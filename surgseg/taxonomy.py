"""Unified class taxonomy shared by data prep, training, evaluation and the VR runtime."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_TAXONOMY = Path(__file__).resolve().parent.parent / "configs" / "taxonomy.yaml"

BACKGROUND = 0
IGNORE = 255
GROUP_BASE = 100
PARTIAL_BASE = 150   # 150 + class id = "this pixel is that class, class not exhaustively labeled"


@dataclass
class Taxonomy:
    names: list[str]                       # index = unified class id
    colors: np.ndarray                     # (C, 3) uint8
    groups: dict[int, tuple[str, list[int]]]   # group id -> (name, member class ids)
    dataset_maps: dict[str, dict]          # raw per-dataset mapping from the YAML

    @property
    def num_classes(self) -> int:
        return len(self.names)

    def class_id(self, name: str) -> int:
        return self.names.index(name)

    def resolve(self, target: str) -> int:
        """Unified name ('liver', 'gi_tract', 'background', 'ignore') -> mask value."""
        if target == "ignore":
            return IGNORE
        if target.startswith("partial:"):
            return PARTIAL_BASE + self.class_id(target[len("partial:"):])
        if target in self.names:
            return self.names.index(target)
        for gid, (gname, _) in self.groups.items():
            if gname == target:
                return gid
        raise KeyError(f"'{target}' is not a unified class or group in the taxonomy")

    def labeled_set(self, dataset: str) -> list[int]:
        """Unified classes (excluding background) a dataset labels exhaustively."""
        ids = set()
        for src, dst in self.dataset_maps[dataset].items():
            if src.startswith("__"):
                continue
            v = self.resolve(dst)
            if 0 < v < GROUP_BASE:
                ids.add(v)
        if dataset == "sisvse":   # instrument parts are matched by suffix
            ids.add(self.class_id("instrument"))
        return sorted(ids)

    def allowed_matrix(self, labeled: list[int]) -> np.ndarray:
        """(256, C) bool: for each mask value, which unified classes it permits.

        This is the heart of partial-label training: value 0 permits background
        plus every class the sample did NOT label; a class id permits only itself;
        a group id permits its members; 255 permits nothing (ignored).
        """
        C = self.num_classes
        m = np.zeros((256, C), dtype=bool)
        m[0, :] = True
        m[0, labeled] = False
        m[0, BACKGROUND] = True
        for c in range(1, C):
            m[c, c] = True
        for gid, (_, members) in self.groups.items():
            m[gid, members] = True
        for c in range(1, C):
            m[PARTIAL_BASE + c, c] = True
        return m

    def value_class(self, v: int) -> int | None:
        """The single unified class a mask value asserts, if any (not bg/groups/ignore)."""
        if 0 < v < self.num_classes:
            return v
        if PARTIAL_BASE < v < PARTIAL_BASE + self.num_classes:
            return v - PARTIAL_BASE
        return None

    def value_name(self, v: int) -> str:
        """Human-readable name of a processed-mask value."""
        if v == IGNORE:
            return "ignore"
        if v in self.groups:
            return self.groups[v][0]
        if PARTIAL_BASE <= v < PARTIAL_BASE + self.num_classes:
            return "partial:" + self.names[v - PARTIAL_BASE]
        return self.names[v]


def load_taxonomy(path: str | Path = DEFAULT_TAXONOMY) -> Taxonomy:
    cfg = yaml.safe_load(Path(path).read_text())
    ids = sorted(int(k) for k in cfg["classes"])
    assert ids == list(range(len(ids))), "class ids must be contiguous from 0"
    assert len(ids) <= GROUP_BASE and len(ids) <= IGNORE - PARTIAL_BASE
    names = [cfg["classes"][i][0] for i in ids]
    colors = np.array([cfg["classes"][i][1] for i in ids], dtype=np.uint8)
    groups = {}
    for gid, (gname, members) in cfg.get("groups", {}).items():
        gid = int(gid)
        assert GROUP_BASE <= gid < PARTIAL_BASE
        groups[gid] = (gname, [names.index(m) for m in members])
    return Taxonomy(names, colors, groups, cfg.get("datasets", {}))
