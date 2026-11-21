"""Multi-dataset, partial-label training. Single GPU or DDP via torchrun.

  torchrun --nproc_per_node=4 -m surgseg.train --config configs/train_student_b2.yaml

Iteration-based; checkpoints to <out>/last.pt and resumes from it automatically, so a
stopped or interrupted run can simply be started again with the same command.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

from surgseg.data import (InfiniteWeightedSampler, SegDataset, available_datasets, collate_eval,
                          load_records, worker_init)
from surgseg.losses import PartialLabelLoss, distillation_loss
from surgseg.metrics import PartialConfusion
from surgseg.model import build_model, load_for_inference, save_checkpoint
from surgseg.taxonomy import load_taxonomy


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world, device


def selection_score(res: dict, cfg: dict) -> tuple[float, float | None]:
    """Score used to pick best.pt.

    By default the mean over datasets. With `select: {classes: [...], weight: w}` it blends
    in the mean IoU of those (danger) classes: rare thin structures peak at a different
    step than the average, and for a safety system they are what matters.
    """
    sel = cfg.get("select") or {}
    mean = res["mean_over_datasets"]
    if not sel.get("classes"):
        return mean, None
    vals = [v["per_class"][c] for v in res.values() if isinstance(v, dict)
            for c in sel["classes"] if c in v.get("per_class", {})]
    danger = sum(vals) / len(vals) if vals else 0.0
    w = sel.get("weight", 0.5)
    return (1 - w) * mean + w * danger, danger


def lr_at(step, cfg):
    o = cfg["optim"]
    if step < o["warmup_iters"]:
        return o["lr"] * (step + 1) / o["warmup_iters"]
    t = (step - o["warmup_iters"]) / max(1, cfg["iters"] - o["warmup_iters"])
    return o["lr"] * (1 - t) ** 1.0 + o.get("min_lr", 0.0) * t   # poly(1.0)


@torch.no_grad()
def validate(model, loader, tax, dataset_names, device, amp_dtype):
    model.eval()
    conf = PartialConfusion(tax.num_classes, len(dataset_names), device)
    for x, m, lab, ds in loader:
        x, m, lab, ds = x.to(device), m.to(device), lab.to(device), ds.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        conf.update(logits.argmax(1), m, lab, ds)
    conf.all_reduce()
    model.train()
    return conf.summary(tax.names, dataset_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[],
                    help="dotted overrides, e.g. iters=200 data.batch_size=2")
    ap.add_argument("--print-out-dir", action="store_true",
                    help="print the resolved out_dir and exit (used by scripts/linux/train.sh)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for ov in args.override:
        k, v = ov.split("=", 1)
        node = cfg
        *path, last = k.split(".")
        for p in path:
            node = node[p]
        val = yaml.safe_load(v)
        if isinstance(val, str):       # PyYAML reads "1e-3" as a string (wants "1.0e-3")
            try:
                val = float(val)
            except ValueError:
                pass
        node[last] = val

    if args.print_out_dir:
        print(os.path.expandvars(cfg["out_dir"]))
        return
    rank, world, device = setup_dist()
    torch.manual_seed(cfg.get("seed", 0) + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(os.path.expandvars(cfg["out_dir"]))
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(cfg))

    tax = load_taxonomy()
    dcfg = cfg["data"]
    root = os.path.expandvars(dcfg["root"])
    dataset_names = available_datasets(root, list(dcfg["datasets"]), verbose=rank == 0)
    kd = cfg.get("distill") or {}
    use_teacher = bool(kd.get("teacher_ckpt"))
    # Unlabeled frames only carry a signal through the teacher; skip them without one.
    train_recs = load_records(root, dataset_names, "train", unlabeled=use_teacher)
    val_recs = load_records(root, dataset_names, "val", unlabeled=False)
    if rank == 0:
        n_unl = sum(r.get("mask") is None for r in train_recs)
        print(f"train {len(train_recs)} ({n_unl} unlabeled) / val {len(val_recs)} samples "
              f"from {dataset_names}")

    train_ds = SegDataset(root, train_recs, tax.num_classes, dataset_names,
                          crop=dcfg["crop"], train=True)
    boost = dcfg.get("boost") or {}
    sampler = InfiniteWeightedSampler(train_recs, dcfg["datasets"], cfg.get("seed", 0), rank,
                                      dcfg.get("synthetic_weight", 1.0),
                                      [tax.class_id(c) for c in boost.get("classes", [])],
                                      boost.get("factor", 1.0))
    train_loader = DataLoader(train_ds, batch_size=dcfg["batch_size"], sampler=sampler,
                              num_workers=dcfg["workers"], pin_memory=True, drop_last=True,
                              persistent_workers=dcfg["workers"] > 0, worker_init_fn=worker_init,
                              prefetch_factor=4 if dcfg["workers"] > 0 else None)
    val_ds = SegDataset(root, val_recs[rank::world], tax.num_classes, dataset_names,
                        train=False, eval_long_side=dcfg.get("eval_long_side", 640))
    val_loader = DataLoader(val_ds, batch_size=dcfg.get("eval_batch_size", 8),
                            num_workers=dcfg.get("eval_workers", 2), collate_fn=collate_eval,
                            worker_init_fn=worker_init)

    mcfg = cfg["model"]
    model = build_model(mcfg["preset"], tax.num_classes, pretrained=mcfg.get("pretrained", True),
                        **mcfg.get("overrides", {})).to(device)
    if mcfg.get("init_ckpt") and not (out / "last.pt").exists():
        init = torch.load(os.path.expandvars(mcfg["init_ckpt"]), map_location=device,
                          weights_only=False)
        model.load_state_dict(init["model"])
        if rank == 0:
            print(f"initialised from {mcfg['init_ckpt']}")
    if world > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]
                                                          if device.type == "cuda" else None)
    # EMA of the weights: a cheap, reliable accuracy gain; it is what gets validated/exported.
    ema = None
    if cfg.get("ema_decay"):
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model.module if world > 1 else model, use_buffers=True,
                            multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]))
    # One or several teachers; several = ensemble (average of softened probabilities).
    teachers = []
    if use_teacher:
        tc = kd["teacher_ckpt"]
        for path in ([tc] if isinstance(tc, str) else tc):
            t_model, _ = load_for_inference(os.path.expandvars(path), device)
            for p in t_model.parameters():
                p.requires_grad_(False)
            teachers.append(t_model)

    criterion = PartialLabelLoss(tax, cfg["loss"].get("dice_weight", 0.5),
                                 cfg["loss"].get("class_weights")).to(device)
    o = cfg["optim"]
    # No weight decay on norms/biases; optionally a lower LR for the pretrained encoder.
    enc_ids = {id(p) for p in (model.module if world > 1 else model).encoder.parameters()}
    buckets: dict[tuple[float, float], list] = {}
    for n, p in model.named_parameters():
        wd = 0.0 if p.ndim <= 1 or n.endswith(".bias") else o["weight_decay"]
        mult = o.get("encoder_lr_mult", 1.0) if id(p) in enc_ids else 1.0
        buckets.setdefault((wd, mult), []).append(p)
    optim = torch.optim.AdamW([{"params": ps, "weight_decay": wd, "lr_mult": mult}
                               for (wd, mult), ps in buckets.items()], lr=o["lr"])
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.get("amp", "bf16"))
    if device.type != "cuda":
        amp_dtype = None
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    step, best = 0, -1.0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        (model.module if world > 1 else model).load_state_dict(ck["raw_model"] or ck["model"])
        if ema is not None and ck.get("ema"):
            ema.module.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optimizer"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        step, best = ck["step"], ck["best"]
        if rank == 0:
            print(f"resumed from step {step} (best {best:.4f})")

    log = open(out / "log.jsonl", "a") if rank == 0 else None
    it = iter(train_loader)
    t0 = time.time()
    model.train()
    while step < cfg["iters"]:
        x, m, lab, _ = next(it)
        x, m, lab = (x.to(device, non_blocking=True), m.to(device, non_blocking=True),
                     lab.to(device, non_blocking=True))
        lr = lr_at(step, cfg)
        for g in optim.param_groups:
            g["lr"] = lr * g["lr_mult"]
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
            if teachers:
                with torch.no_grad():
                    T = kd.get("T", 2.0)
                    t_probs = sum(torch.softmax(t(x).float() / T, 1) for t in teachers)
                    t_probs = t_probs / len(teachers)
        losses = criterion(logits, m, lab)
        loss = losses["loss"]
        if teachers:
            # KD on EVERY pixel, including unlabeled frames - that is where the extra data
            # (thousands of unannotated surgical frames) teaches the student.
            losses["kd"] = distillation_loss(logits, t_probs, torch.ones_like(m, dtype=torch.bool),
                                             kd.get("T", 2.0))
            loss = loss + kd.get("weight", 1.0) * losses["kd"]
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), o.get("clip_grad", 1.0))
        scaler.step(optim)
        scaler.update()
        if ema is not None:
            ema.update_parameters(model.module if world > 1 else model)
        step += 1
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"non-finite loss at step {step}: {losses}")

        if rank == 0 and step % cfg.get("log_every", 50) == 0:
            rec = {"step": step, "lr": lr, **{k: round(v.item(), 4) for k, v in losses.items()},
                   "it_per_s": round(cfg.get("log_every", 50) / (time.time() - t0), 2)}
            t0 = time.time()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()

        is_val = step % cfg["val_every"] == 0 or step == cfg["iters"]
        if is_val:
            res = validate(ema.module if ema is not None else model, val_loader, tax,
                           dataset_names, device, amp_dtype)
            score, danger = selection_score(res, cfg)
            if rank == 0:
                print(json.dumps({"step": step, "val": res}, indent=1), flush=True)
                print(json.dumps({"step": step, "select_score": round(score, 4),
                                  "danger_mIoU": danger}), flush=True)
                log.write(json.dumps({"step": step, "val": res, "select_score": score,
                                      "danger_mIoU": danger}) + "\n")
                log.flush()
                if score > best:
                    best = score
                    save_checkpoint(out / "best.pt", model, None, None, step, best, cfg,
                                    tax.names, ema)
        # Frequent resumable checkpoints: a requeued/preempted job loses little work.
        if is_val or step % cfg.get("ckpt_every", 1000) == 0:
            if rank == 0:
                save_checkpoint(out / "last.pt.tmp", model, optim, scaler.state_dict(), step,
                                best, cfg, tax.names, ema)
                os.replace(out / "last.pt.tmp", out / "last.pt")   # atomic: safe vs preemption
            if world > 1:
                dist.barrier()
    if rank == 0:   # scripts/linux/ only lets later stages start once this exists
        (out / "DONE").write_text(f"finished {step} iterations\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
