# SurgSeg: How the Model Was Trained, and What the Project Is Trying to Do

*Last updated: 21 October 2026*

---

## 1. What the project is trying to do

**Goal:** add an AI safety layer to da Vinci VR surgical training that works in **any simulator**.

The system has these parts:

1. **Simulator link.** A Unity VR simulator (CRESSim + an organ scene built from BodyParts3D) sends each camera frame, its depth and the instrument tip positions to a model server.
2. **Segmentation.** The model labels every pixel with one of 22 tissue classes (liver, gallbladder, duct, artery, instrument, ...).
3. **Two modes for target and danger tissue:**
   - **Auto:** a procedure profile assigns roles, e.g. cholecystectomy: gallbladder = target; cystic duct and artery = danger.
   - **Manual:** the trainee or instructor clicks or draws target and danger regions. The model snaps each mark to the real structure.
4. **Warnings.** The distance from each instrument tip to danger tissue gives OK, WARNING or VIOLATION.
5. **Locked overlay.** Segmentation runs at about 1 Hz, but the highlights are re-projected with depth every frame, so they stay on the tissue as the head or camera moves.

**Role of the model:** it answers *"what tissue is this?"* from the image alone. The procedure rules or the user answer *"is it target or danger?"*. Because it reads only images, the same model works in simulators whose meshes are not labelled, and on real surgical video. Auto mode depends on it.

**Research focus (proposed):** tracking, meaning keeping marks on the right tissue while it deforms, is hidden by tools or is cut. No published method solves this. A simulator gives exact per-frame ground truth that real surgery cannot. See section 6.

---

## 2. Datasets

Every dataset labels a different subset of anatomy. All were mapped into **one 22-class taxonomy** (`configs/taxonomy.yaml`).

| Dataset | Procedure / content | What it contributes | Frames we prepared | Licence |
|---|---|---|---|---|
| **SISVSE** [1] | da Vinci robotic gastrectomy, 40 cases | Liver, stomach, pancreas, gallbladder, spleen, many da Vinci instrument types | **14,872** (real frames only; its synthetic splits were not in our download and were skipped) | BSD-3 |
| **CholecSeg8k** [2] | Laparoscopic cholecystectomy (Cholec80) | Liver, gallbladder, fat, connective tissue, abdominal wall, cystic duct, hepatic vein, blood | **8,080** | CC BY-NC-SA |
| **Endoscapes2023** [3] | Laparoscopic cholecystectomy, 201 videos | **Cystic artery and cystic duct** (the key danger structures), gallbladder, cystic plate, Calot's triangle; ~58k unlabelled frames | **493** labelled frames (its segmentation subset), plus unlabelled frames for distillation | CC BY-NC-SA |
| **DSAD** [4] | da Vinci rectal resection | 11 structures incl. **ureter**, inferior mesenteric artery, veins, colon, pancreas, spleen | Present (in validation and test results) | CC BY 4.0 |
| **SAR-RARP50** [5] | da Vinci Si prostatectomy (suturing) | Instrument parts, needle, thread, clamps; unlabelled video frames | Present (in validation and test results) | CC BY-NC-SA |
| **m2caiSeg** [6] | Laparoscopic cholecystectomy, 19 classes (small) | Artery, liver, gallbladder, fat, instruments | Present (weight 0.3; in test results) | CC BY-NC-SA |
| EndoVis 2017 / 2018 [7, 8] | da Vinci instruments / nephrectomy scene | Instruments, kidney | Needs grand-challenge sign-up; not confirmed, used automatically if present | Challenge terms |
| **Our simulator (randomized)** | Unity organ scene, domain-randomized | Exact masks for liver, gallbladder, stomach, duodenum, duct, artery, fat backdrop, instrument | **800 train + 120 val** at 640x512 | Ours |

To see which datasets were prepared: `ls $SURGSEG_DATA/processed`

**Licence note:** several datasets and the SurgeNetXL weights are non-commercial, so the trained model is for research and education only.

---

## 3. How the datasets were combined

**Unified taxonomy (22 classes):** background, instrument, liver, gallbladder, stomach, small intestine, colon, pancreas, spleen, kidney, ureter, vesicular glands, artery, vein, duct, fat, connective tissue, abdominal wall, blood, suture, surgical object, uterus.

**Problem:** datasets contradict each other. CholecSeg8k labels fat, but Endoscapes calls fat "background", so naive training would teach the model that fat is background.

**Solution: partial-label ("marginal") loss** [9], extended to groups:

- When dataset X calls a pixel "background", the model learns only "none of the classes X labels", never "definitely background".
- **Groups** handle ambiguous labels. CholecSeg8k's "gastrointestinal tract" means stomach OR small intestine OR colon, and the loss uses the sum of those probabilities.
- **Partial labels** handle non-exhaustive labels. Endoscapes' cystic plate is connective tissue, but Endoscapes does not label all connective tissue, so its background is not a negative.
- Source labels that are not anatomy (spilled bile, "unknown") are **ignored**.

**Safety-focused sampling:** frames that show artery, vein, duct, ureter or vesicular glands are drawn 2-3x more often, and those classes get 1.5x loss weight. The checkpoint is chosen on overall mIoU **and** on the danger classes.

---

## 4. Training pipeline

Trained on a Linux machine with 4x NVIDIA A100 80 GB GPUs (`scripts/linux/`; runs resume automatically from `last.pt`).

```
LemonFM teacher (large, accurate)  ->  SurgeNetXL student (small, fast)  ->  calibrate  ->  ONNX
                                                                       ->  fine-tune on simulator frames
```

| Stage | Model | Key settings | Config |
|---|---|---|---|
| **1. Teacher** | **LemonFM** backbone [10] (ConvNeXt-Large [11], self-supervised on 938 h / 4k surgical videos) + **UPerNet** head [12], ~230M params | 60k iters, 768 crops, bf16, EMA 0.9998, lr 1e-4 (backbone x0.1), CE + 0.5 Dice, partial-label loss | `train_teacher_lemonfm.yaml` |
| **2. Student (deployed)** | **SurgeNetXL** backbone [13] (CaFormer-S18 [14], self-supervised on 4.7M surgical frames) + **FPN** head [15], ~28M params | Knowledge distillation [16] from the teacher (T = 2, weight 1.0) on labelled + **unlabelled** Endoscapes and SAR-RARP50 frames; 40k iters, 640 crops, batch 12/GPU; deployed checkpoint = step 28,000 | `train_student_kd.yaml` |
| **3. Calibration + export** | Student | Temperature scaling [17] fitted on validation (T = 1.34), baked into ONNX; exported at 512x640 / 640x800 for the VR server | `scripts/linux/eval_export.sh` |
| **4. Sim fine-tune** | Student, initialized from stage 2 | 800 randomized sim frames (weight 3.0) **mixed with real data** so real performance is kept; danger classes weighted 3x; 15k iters, lr 5e-5 | `finetune_sim.yaml` |

The pipeline also supports a 3-teacher ensemble, pseudo-label completion and a round-2 teacher (`run_pipeline.sh`). The deployed student was distilled from the round-1 teacher (`teacher_lemonfm_s0`).

**Simulator data (stage 4):** rendered in Unity batch mode by `SurgSegDatasetGenerator.cs`. Each frame randomizes camera pose, FOV, lighting, tissue colours and textures, backdrop, organ pose (+/-15 deg, +/-10% scale) and instrument pose. This is **domain randomization** [18], so the model learns anatomy and context, not one simulator's look. Masks come directly from the engine and are exact.

---

## 5. Current accuracy

| Model / data | Result |
|---|---|
| Teacher, real validation (mean over datasets, step 28k to 60k) | **mIoU about 0.63 to 0.64**; danger classes about 0.36 (Endoscapes duct 0.56, artery 0.17; DSAD ureter 0.28) |
| Deployed student, real **test** | **Mean mIoU 0.622** over 5 datasets: DSAD 0.670, CholecSeg8k 0.506, Endoscapes 0.569, SAR-RARP50 0.690, m2caiSeg 0.675 |
| Deployed student, our **simulator** frames (before fine-tune) | mIoU **0.14**; artery and duct **0.00** |
| After sim fine-tune, real test | Mean over the same 4 datasets 0.609 -> **0.602** (real accuracy kept); Endoscapes duct 0.37 -> 0.29, artery 0.29 -> 0.26. Accuracy on simulator frames not yet measured |
| Point tracker (LiteTracker [19]) on STIR 2024 [20], our laptop | **75.7** delta-avg (paper: 75.8), about 18-30 ms per frame |

**Weak point:** thin danger structures (artery, duct, ureter) are the hardest classes, yet they matter most.

---

## 6. What is new, and the research gap

**Not novel by itself:** the segmentation model. Surgical anatomy segmentation is established (e.g. GoNoGoNet, CVS models), and our training techniques are known.

**What is new:**

1. Anatomy-aware danger guidance **inside robotic VR training**, learned from real surgery. Simulators and AI tutors judge motion; danger-zone AI has only been used on real OR video.
2. **Simulator-agnostic:** uses only image, depth and tool positions, so it moves to other simulators and to real video.
3. **Manual vs auto** definition of danger in one system, enabling the first comparison for learning.
4. **One model for sim and real**, via domain randomization plus real-data co-training.
5. **(Research direction) Tracking:** simulator ground truth for tissue tracking through deformation, occlusion and cutting, plus a tracker that uses segmentation to stay on the right tissue. First demo: danger precision went from 31% to 99%.

**Research gap:** keeping marks on moving tissue. On STIR, a 2023 baseline still beats every newer method, no method tracks natively in 3D, and none handles cutting. The root cause is missing ground truth, which a simulator can provide.

Closest related simulator work: **SISVSE** [1] also built a Unity virtual surgery environment, to make synthetic **training data** for segmentation. We use the simulator the other way round: as the **training environment for surgeons**, with segmentation guiding them.

---

## 7. References

**Datasets**

1. Yoon J., Choi M.-K. et al. *Surgical Scene Segmentation Using Semantic Image Synthesis with a Virtual Surgery Environment* (SISVSE). MICCAI 2022, LNCS 13437, pp. 551-561. https://link.springer.com/10.1007/978-3-031-16449-1_53
2. Hong W.-Y. et al. *CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic Cholecystectomy Based on Cholec80.* arXiv:2012.12453, 2020.
3. Murali A. et al. *Endoscapes: A Critical View of Safety and Surgical Scene Segmentation Dataset for Laparoscopic Cholecystectomy.* arXiv:2312.12429, 2023.
4. Carstens M. et al. *The Dresden Surgical Anatomy Dataset for Abdominal Organ Segmentation in Surgical Data Science* (DSAD). Scientific Data 10, 2023.
5. Psychogyios D. et al. *SAR-RARP50: Segmentation of Surgical Instrumentation and Action Recognition on Robot-Assisted Radical Prostatectomy Challenge.* arXiv:2401.00496, 2023.
6. Maqbool S. et al. *m2caiSeg: Semantic Segmentation of Laparoscopic Images using Convolutional Neural Networks.* arXiv:2008.10134, 2020.
7. Allan M. et al. *2017 Robotic Instrument Segmentation Challenge.* arXiv:1902.06426, 2019.
8. Allan M. et al. *2018 Robotic Scene Segmentation Challenge.* arXiv:2001.11190, 2020.

**Methods and models**

9. Shi G., Xiao L., Chen Y., Zhou S. K. *Marginal Loss and Exclusion Loss for Partially Supervised Multi-Organ Segmentation.* Medical Image Analysis 70, 2021. arXiv:2007.03868.
10. Che C. et al. *LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings* (LemonFM). arXiv:2503.19740, 2025 (CVPR 2026).
11. Liu Z. et al. *A ConvNet for the 2020s* (ConvNeXt). CVPR 2022. arXiv:2201.03545.
12. Xiao T. et al. *Unified Perceptual Parsing for Scene Understanding* (UPerNet). ECCV 2018. arXiv:1807.10221.
13. Jaspers T. J. M. et al. *Scaling up Self-Supervised Learning for Improved Surgical Foundation Models* (SurgeNetXL). arXiv:2501.09436, 2025.
14. Yu W. et al. *MetaFormer Baselines for Vision* (CaFormer). arXiv:2210.13452, 2022.
15. Lin T.-Y. et al. *Feature Pyramid Networks for Object Detection* (FPN). CVPR 2017. arXiv:1612.03144.
16. Hinton G., Vinyals O., Dean J. *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531, 2015.
17. Guo C. et al. *On Calibration of Modern Neural Networks* (temperature scaling). ICML 2017. arXiv:1706.04599.
18. Tobin J. et al. *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.* IROS 2017. arXiv:1703.06907.

**Tracking and simulator**

19. Karaoglu M. A. et al. *LiteTracker: Leveraging Temporal Causality for Accurate Low-Latency Tissue Tracking.* MICCAI 2025. arXiv:2504.09904.
20. Schmidt A. et al. *Point Tracking in Surgery: The 2024 Surgical Tattoos in Infrared (STIR) Challenge.* arXiv:2503.24306; *STIRC2025*, arXiv:2607.12939.
21. Ou Y. et al. *A Realistic Surgical Simulator for Non-Rigid and Contact-Rich Manipulation in Surgeries with the da Vinci Research Kit* (CRESSim). UR 2024. arXiv:2404.05888.
22. Mitsuhashi N. et al. *BodyParts3D: 3D Structure Database for Anatomical Concepts.* Nucleic Acids Research 37, 2009.

Full literature review (related systems, novelty, evaluation plan): https://claude.ai/artifact/PR4Ao7AVe83ptszZ45co5j
