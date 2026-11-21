"""PyTorch datasets, augmentation and the multi-dataset sampler."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def available_datasets(data_root: str | Path, names: list[str], verbose: bool = True) -> list[str]:
    """Datasets that were actually prepared (EndoVis etc. need manual sign-up)."""
    have = [d for d in names if (Path(data_root) / d / "index.jsonl").exists()]
    missing = [d for d in names if d not in have]
    if missing and verbose:
        print(f"WARNING: skipping datasets that are not prepared: {missing}")
    if not have:
        raise FileNotFoundError(f"none of {names} found under {data_root}")
    return have


def load_records(data_root: str | Path, datasets: list[str], split: str,
                 unlabeled: bool = True) -> list[dict]:
    """Records of one split. unlabeled=False drops frames without masks (eval, no teacher)."""
    recs = []
    for d in datasets:
        idx = Path(data_root) / d / "index.jsonl"
        if not idx.exists():
            raise FileNotFoundError(f"{idx} missing - run the {d} converter first")
        with open(idx) as f:
            recs += [r for r in map(json.loads, filter(str.strip, f)) if r["split"] == split
                     and (unlabeled or r.get("mask") is not None)]
    return recs


def train_transforms(crop: tuple[int, int]):
    geometric = v2.Compose([
        v2.RandomResizedCrop(crop, scale=(0.2, 1.0), ratio=(0.6, 1.7), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.RandomRotation(25, fill={tv_tensors.Image: 0, tv_tensors.Mask: 255})],
                       p=0.3),
    ])
    # Photometric augmentation on the image only. Fairly strong on purpose: the model has
    # to survive the jump from real endoscope footage to rendered VR tissue.
    photometric = v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.04)], p=0.8),
        v2.RandomApply([v2.GaussianBlur(7, sigma=(0.1, 2.5))], p=0.25),
        v2.RandomAdjustSharpness(2.0, p=0.2),
        v2.RandomGrayscale(p=0.02),
    ])
    return geometric, photometric


def to_input(img_uint8_chw: torch.Tensor) -> torch.Tensor:
    x = img_uint8_chw.float().div_(255)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


def pad_to_multiple(x: torch.Tensor, m: torch.Tensor, k: int = 32):
    h, w = x.shape[-2:]
    ph, pw = (-h) % k, (-w) % k
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), value=0.0)
        m = torch.nn.functional.pad(m, (0, pw, 0, ph), value=255)
    return x, m


class SegDataset(Dataset):
    """Returns (image float CxHxW, mask long HxW, labeled bool vector over classes, ds idx)."""

    def __init__(self, data_root, records, num_classes, dataset_names,
                 crop=(512, 512), train=True, eval_long_side=640):
        self.root = Path(data_root)
        self.records = records
        self.C = num_classes
        self.ds_index = {d: i for i, d in enumerate(dataset_names)}
        self.train = train
        self.eval_long_side = eval_long_side
        if train:
            self.geometric, self.photometric = train_transforms(tuple(crop))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = cv2.cvtColor(cv2.imread(str(self.root / r["image"])), cv2.COLOR_BGR2RGB)
        if r.get("mask"):
            msk = cv2.imread(str(self.root / r["mask"]), cv2.IMREAD_UNCHANGED)
        else:   # unlabeled frame: nothing supervised by labels, teacher distillation only
            msk = np.full(img.shape[:2], 255, np.uint8)
        if not self.train:
            s = self.eval_long_side / max(img.shape[:2])
            size = (round(img.shape[1] * s), round(img.shape[0] * s))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, size, interpolation=cv2.INTER_NEAREST)
        img_t = tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1).contiguous())
        msk_t = tv_tensors.Mask(torch.from_numpy(msk))
        if self.train:
            img_t, msk_t = self.geometric(img_t, msk_t)
            img_t = self.photometric(img_t)
        x = to_input(torch.as_tensor(img_t))
        m = torch.as_tensor(msk_t).long()
        if not self.train:
            x, m = pad_to_multiple(x, m)
        labeled = torch.zeros(self.C, dtype=torch.bool)
        labeled[r["labeled"]] = True
        return x, m, labeled, self.ds_index[r["dataset"]]


class InfiniteWeightedSampler(torch.utils.data.Sampler):
    """Samples so each dataset gets `weight` share of batches regardless of its size.

    Each DDP rank uses its own seed, so ranks see different samples without coordination.
    """

    def __init__(self, records, dataset_weights: dict[str, float], seed: int, rank: int,
                 synthetic_weight: float = 1.0, boost_classes: list[int] | None = None,
                 boost_factor: float = 1.0):
        """boost_*: frames showing a rare, safety-critical class (artery, duct, ureter...)
        are drawn boost_factor times more often within their dataset."""
        boost = set(boost_classes or [])

        def rel(r):
            sw = synthetic_weight if r.get("synthetic") else 1.0
            return sw * (boost_factor if boost & set(r.get("present", [])) else 1.0)

        totals: dict[str, float] = {}
        for r in records:
            totals[r["dataset"]] = totals.get(r["dataset"], 0.0) + rel(r)
        w = [dataset_weights.get(r["dataset"], 1.0) * rel(r) / totals[r["dataset"]]
             for r in records]
        self.weights = torch.tensor(w, dtype=torch.double)
        self.g = torch.Generator().manual_seed(seed * 1000 + rank)

    def __iter__(self):
        while True:
            yield from torch.multinomial(self.weights, 65536, replacement=True,
                                         generator=self.g).tolist()


def worker_init(_):
    cv2.setNumThreads(0)


def collate_eval(batch):
    """Eval images keep their aspect ratio, so pad the batch to a common size."""
    H = max(b[0].shape[1] for b in batch)
    W = max(b[0].shape[2] for b in batch)
    xs, ms = [], []
    for x, m, *_ in batch:
        ph, pw = H - x.shape[1], W - x.shape[2]
        xs.append(torch.nn.functional.pad(x, (0, pw, 0, ph)))
        ms.append(torch.nn.functional.pad(m, (0, pw, 0, ph), value=255))
    return (torch.stack(xs), torch.stack(ms), torch.stack([b[2] for b in batch]),
            torch.tensor([b[3] for b in batch]))
