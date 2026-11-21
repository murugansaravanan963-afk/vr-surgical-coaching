# Code guide: what every file does

The repo looks big, but it has only **three jobs**:

1. **Prepare data**: turn 6 differently-labelled public datasets into one format.
2. **Train the model**: teacher → student → calibrate → export ONNX.
3. **Run it live**: in every frame, segment → assign target/danger → check tool distance → show it in VR.

Most files are small helpers. The **core ideas live in about 8 files**, marked ★ below.
Understand those and you understand the project.

---

## 1. Prepare data

| File | What it does, in one or two lines |
|---|---|
| ★ [configs/taxonomy.yaml](../configs/taxonomy.yaml) | The **one list of 22 classes** (liver, gallbladder, duct, artery, instrument, ...) and, for each dataset, how its own label names map onto those 22. Also defines **groups** (e.g. "GI tract" = stomach, small intestine or colon) for datasets that label more coarsely. |
| [surgseg/taxonomy.py](../surgseg/taxonomy.py) | Loads that YAML. Builds the "allowed classes" table the loss uses: for each mask value, which of the 22 classes are possible there. |
| [surgseg/datasets/common.py](../surgseg/datasets/common.py) | Shared by all converters. Writes every dataset in the same layout (`images/`, `masks/`, `index.jsonl`) and splits train/val/test **by video**, never by frame, because frames from one video are near-duplicates and would leak into the test set. |
| `surgseg/datasets/prepare_*.py` (10 files) | One converter per dataset. They all do the same thing: read that dataset's format, map its labels through the taxonomy, write the common format. The differences are only file layouts (colour-coded masks, JSON polygons, parquet files, video frames). |
| [surgseg/datasets/prepare_custom.py](../surgseg/datasets/prepare_custom.py) | Same, for frames rendered by **our Unity simulator** (exact masks come from the engine). |
| [surgseg/datasets/prepare_unlabeled.py](../surgseg/datasets/prepare_unlabeled.py) | Extracts frames from unlabelled surgical video. The student learns from the teacher's predictions on them (no masks needed). |
| [surgseg/tools/check_data.py](../surgseg/tools/check_data.py) | Sanity check: counts per dataset, no video in two splits, and **preview images with the masks drawn on top**, so a wrong label mapping is visible before training. |
| [surgseg/tools/refine_index.py](../surgseg/tools/refine_index.py) | Re-marks which frames contain thin rare classes (a visible duct is tiny) and can re-pick validation videos so they actually contain duct/vein. |

## 2. Train the model

| File | What it does |
|---|---|
| ★ [surgseg/losses.py](../surgseg/losses.py) | **Partial-label loss**, the key trick. If dataset X says a pixel is "background", that only means "none of the classes X labels". So the loss sums the probability of every class that is still allowed there instead of forcing "background". Also Dice loss, extra weight for danger classes, and the **distillation loss** (student copies the teacher's softened probabilities). |
| ★ [surgseg/train.py](../surgseg/train.py) | The training loop. Reads a config, builds the model, mixes datasets, trains for N iterations with warm-up + poly learning rate, mixed precision (bf16), EMA weights; validates per dataset; keeps `best.pt` by mean IoU (optionally blended with danger-class IoU). Saves `last.pt` regularly, so a stopped run resumes. Works on 1 GPU or many (torchrun). |
| [surgseg/data.py](../surgseg/data.py) | Loads images and masks, augmentations (crop, flip, colour jitter...). **InfiniteWeightedSampler** gives each dataset a fixed share of every batch regardless of its size, and draws frames that show rare danger classes more often (2× in the deployed model's config). |
| [surgseg/model.py](../surgseg/model.py) | Builds the network (backbone + decoder via `segmentation_models_pytorch`), saves and loads checkpoints. |
| [surgseg/backbones.py](../surgseg/backbones.py) | Downloads the **surgical foundation backbones** (LemonFM for the teacher, SurgeNetXL for the student) and converts their weights to load correctly. These were pretrained on surgical video, not ImageNet, which helps a lot. |
| [surgseg/pseudo_label.py](../surgseg/pseudo_label.py) | Optional self-training: an ensemble of teachers fills in missing labels and labels unlabelled frames. *Supported by the pipeline but not used for the deployed model* (see below). |
| ★ [surgseg/evaluate.py](../surgseg/evaluate.py) | Scores a checkpoint per dataset (IoU per class) and **fits a calibration temperature** so the probabilities are honest. This matters because the live system uses thresholds like "danger if P > 0.3". |
| [surgseg/metrics.py](../surgseg/metrics.py) | IoU that respects partial labels: a prediction of a class a dataset never labels is not counted as wrong for that dataset. |
| [surgseg/export_onnx.py](../surgseg/export_onnx.py) | Exports the model to **ONNX** with normalisation and the calibration temperature baked in, plus `surgseg.json` (class names, colours, input size). Any engine (Python, Unity, Unreal) can run it the same way. |
| `configs/train_*.yaml` | One file per training run: which backbone, datasets and weights, iterations, learning rate, loss settings. `train_teacher_lemonfm.yaml` → teacher; `train_student_kd.yaml` → the deployed student; `finetune_sim.yaml` → simulator fine-tune. `configs/baselines/` = ImageNet SegFormer for comparison. |
| `scripts/linux/*.sh` | Plumbing that runs the steps above in order: `setup_env.sh` (install) → `download_data.sh` → `prepare_data.sh` → `train.sh` / `run_pipeline.sh` → `eval_export.sh`. `env.sh` only sets folder paths. |
| [surgseg/tools/benchmark.py](../surgseg/tools/benchmark.py) | Measures ONNX speed (ms per frame) at given sizes. |
| [surgseg/tools/confusion.py](../surgseg/tools/confusion.py), [scripts/linux/confusion_matrix.py](../scripts/linux/confusion_matrix.py) | "When the truth is duct, what does the model predict?" Used to diagnose weak classes. |
| [surgseg/tools/demo_images.py](../surgseg/tools/demo_images.py) | Runs the model on a folder of images and saves overlay pictures. |

### What was actually used for the deployed model

| Step | Used? |
|---|---|
| 6 datasets: DSAD, SISVSE (real frames), CholecSeg8k, Endoscapes, SAR-RARP50, m2caiSeg | Yes |
| One LemonFM teacher (`teacher_lemonfm_s0`, 60k iterations) | Yes |
| Student distilled from that teacher, on labelled + unlabelled Endoscapes/SAR-RARP50 frames (`student_kd`, checkpoint at step 28k) | **Yes: this is `surgseg.onnx`** |
| Calibration (temperature T = 1.34) + ONNX export | Yes |
| 3-teacher ensemble, pseudo-labels, round-2 teacher | No: the code supports it, but it was not run |
| EndoVis 2017/2018, SISVSE synthetic frames | No: data not available |
| Simulator fine-tune (`student_sim`) | Trained; real accuracy 0.609 → 0.602, so `student_kd` stayed the deployed model |

## 3. Run it live

| File | What it does |
|---|---|
| [surgseg/runtime/engine.py](../surgseg/runtime/engine.py) | Loads `surgseg.onnx` with ONNX Runtime (GPU if available) and returns per-pixel class, probabilities and confidence. No PyTorch needed. |
| ★ [surgseg/runtime/roles.py](../surgseg/runtime/roles.py) | Turns **anatomy into roles**: target, danger, neutral, instrument. **Mode 1 manual**: the user points at tissue (`select(u, v, role)`), either the whole class or just that blob, which is then tracked frame to frame. **Mode 2 auto**: a procedure profile (`configs/roles/*.yaml`) says what is target and what is danger. Safety bias: danger = P(any danger class) ≥ 0.3, not argmax, so uncertain pixels near a vessel err toward warning. One mark is capped at 15% of the view. |
| ★ [surgseg/runtime/safety.py](../surgseg/runtime/safety.py) | For each tool tip: **distance to the nearest danger pixel** (distance transform; in mm when depth is available, else pixels). Levels: **OK** → **WARNING** (within 10 mm / 40 px) → **VIOLATION** (within 2 mm / 6 px). Hysteresis: rises at once, falls only after 3 calm frames, so it does not flicker. Frames that are too old are reported as **STALE** instead of trusted. |
| [surgseg/runtime/server.py](../surgseg/runtime/server.py) | A small **ZeroMQ server**: the simulator sends a frame + tool-tip positions, and gets back role map, alert levels and labels. Any engine with ZeroMQ can use it. |
| [surgseg/runtime/client_example.py](../surgseg/runtime/client_example.py) | A minimal Python client showing the request/response. |
| `configs/roles/*.yaml` | Procedure profiles (cholecystectomy, nephrectomy, rectal resection, gastrectomy, default): target and danger classes, optionally per surgical step. |

### Unity side ([integration/unity/](../integration/unity/))

| File | What it does |
|---|---|
| [SurgSegClient.cs](../integration/unity/SurgSegClient.cs) | Unity client: sends the endoscope image + tool tips to the server (only when the previous result is back), shows the colour overlay and the HUD with each tip's level and distance. |
| ★ [SurgSegWorldLock.cs](../integration/unity/SurgSegWorldLock.cs) + [SurgSegReproject.shader](../integration/unity/SurgSegReproject.shader) + [SurgSegEyeDepth.shader](../integration/unity/SurgSegEyeDepth.shader) | **Keeps the overlay stuck to the tissue** while the head/camera moves. When a frame is sent, its depth and camera pose are saved. Every rendered frame, each visible surface point is projected back into that saved view to look up its colour. Result: no lag or sliding, even though the model is slower than the display. |
| [SurgSegCapture.cs](../integration/unity/SurgSegCapture.cs) + [SurgSegId.shader](../integration/unity/SurgSegId.shader) + [SurgSegLabel.cs](../integration/unity/SurgSegLabel.cs) | Records image + exact label-mask pairs from Unity. `SurgSegLabel` tags each object with its class; the ID shader renders the class number as the mask. |
| [Editor/SurgSegDatasetGenerator.cs](../integration/unity/Editor/SurgSegDatasetGenerator.cs) | Generates a **domain-randomised** training set automatically: random camera, lighting, tissue colours, organ pose, instrument pose, with exact masks. |
| [Editor/SurgSegOrganSceneBuilder.cs](../integration/unity/Editor/SurgSegOrganSceneBuilder.cs) | Builds the test scene from BodyParts3D organ meshes (in `Anatomy/`), adds the endoscope camera, an instrument, and the overlay components. |
| [SurgSegToolController.cs](../integration/unity/SurgSegToolController.cs), [SurgSegCameraOrbit.cs](../integration/unity/SurgSegCameraOrbit.cs) | Keyboard control of the instrument (pivoting through a fixed trocar point, like a real robot tool) and of the camera. |
| `Plugins/NetMQ.dll`, `AsyncIO.dll` | Third-party ZeroMQ library for C#. |

## 4. Tests

[tests/test_core.py](../tests/test_core.py) checks the parts where a silent bug would be dangerous: the partial-label
loss maths, metrics ignoring unlabelled classes, manual and auto role assignment, region tracking,
safety levels and hysteresis, 3-D distances, label completion never contradicting ground truth,
and the rare-class sampler. [tests/test_backbones.py](../tests/test_backbones.py) checks that converted
backbone weights compute the same features as the originals.
