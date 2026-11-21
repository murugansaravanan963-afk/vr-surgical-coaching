"""Turning per-pixel anatomy into TARGET / DANGER regions, in two modes.

MANUAL mode - the trainee/instructor picks regions. In VR they point the controller at
    tissue; the engine projects the hit point into the endoscope camera and sends its
    normalised (u, v). Two scopes:
      scope="class"  : every pixel of that anatomy class gets the role (robust, persists)
      scope="region" : only the picked connected blob, tracked frame-to-frame by overlap
                       (e.g. "this one vessel", not every vessel)

AUTO mode - the system assigns roles from a procedure profile (configs/roles/*.yaml),
    a small knowledge base: which structures are the surgical target and which are
    critical for this procedure and, optionally, this step. Manual picks made while in
    AUTO act as overrides on top of the profile.

Safety bias: DANGER is decided on summed probability with a low threshold
(e.g. P(any danger class) > 0.3), not on argmax, so uncertain pixels near a critical
structure err toward warning. TARGET uses argmax + a higher confidence bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np
import yaml

from surgseg.runtime.engine import SegResult

PROFILES_DIR = Path(__file__).resolve().parents[2] / "configs" / "roles"


class Role(IntEnum):
    NEUTRAL = 0
    TARGET = 1
    DANGER = 2
    INSTRUMENT = 3


ROLE_COLORS = {Role.NEUTRAL: (0, 0, 0), Role.TARGET: (0, 200, 0),
               Role.DANGER: (255, 0, 0), Role.INSTRUMENT: (120, 120, 255)}


@dataclass
class Profile:
    name: str
    target: list[str]
    danger: list[str]
    danger_threshold: float = 0.3
    target_threshold: float = 0.5
    min_region_frac: float = 0.002
    steps: dict = field(default_factory=dict)
    description: str = ""

    def for_step(self, step: str | None) -> tuple[list[str], list[str]]:
        if step and step in self.steps:
            s = self.steps[step]
            return s.get("target", self.target), s.get("danger", self.danger)
        return self.target, self.danger


def load_profiles(directory: str | Path = PROFILES_DIR) -> dict[str, Profile]:
    out = {}
    for p in sorted(Path(directory).glob("*.yaml")):
        d = yaml.safe_load(p.read_text())
        out[d["name"]] = Profile(**d)
    return out


@dataclass
class TrackedRegion:
    class_id: int
    role: Role
    mask: np.ndarray            # bool at model resolution
    lost_frames: int = 0


@dataclass
class RoleOutput:
    role_map: np.ndarray        # (h, w) uint8 of Role values
    class_roles: dict[int, Role]
    regions: list[dict]         # connected TARGET/DANGER blobs for anchored labels


class RoleAssigner:
    def __init__(self, class_names: list[str], profiles: dict[str, Profile] | None = None,
                 mode: str = "auto", procedure: str = "default", region_track_frames: int = 15,
                 max_mark_frac: float = 0.15):
        self.names = class_names
        self.max_mark_frac = max_mark_frac        # one user mark never covers more than this of the view
        self.profiles = profiles if profiles is not None else load_profiles()
        for p in self.profiles.values():          # fail at startup, not mid-session
            lists = [p.target, p.danger] + [s.get(k, []) for s in p.steps.values()
                                             for k in ("target", "danger")]
            bad = {n for lst in lists for n in lst if n not in class_names}
            if bad:
                raise ValueError(f"profile '{p.name}' uses unknown classes {sorted(bad)}")
        self.instrument = class_names.index("instrument")
        self.region_track_frames = region_track_frames
        self.class_overrides: dict[int, Role] = {}
        self.regions: list[TrackedRegion] = []
        self.set_mode(mode, procedure)

    # ----------------------------------------------------------------- configuration
    def set_mode(self, mode: str, procedure: str | None = None, step: str | None = None):
        if mode not in ("auto", "manual"):
            raise ValueError("mode must be 'auto' or 'manual'")
        if procedure is not None and procedure not in self.profiles:
            raise KeyError(f"unknown procedure '{procedure}'; have {list(self.profiles)}")
        proc = procedure if procedure is not None else self.procedure
        if step is not None and step not in self.profiles[proc].steps:
            raise KeyError(f"procedure '{proc}' has no step '{step}'; "
                           f"have {list(self.profiles[proc].steps)}")
        self.mode, self.procedure, self.step = mode, proc, step
        self.clear()

    def clear(self):
        self.class_overrides.clear()
        self.regions.clear()

    def select(self, seg: SegResult, u: float, v: float, role: Role | str,
               scope: str = "class") -> dict:
        """User picks the tissue at normalised (u, v) and gives it a role."""
        role = Role[role.upper()] if isinstance(role, str) else Role(role)
        r, c = seg.to_px(u, v)
        cls = int(seg.labels[r, c])
        if cls == self.instrument:
            return {"ok": False, "reason": "picked an instrument, not tissue"}
        if cls == 0:
            return {"ok": False, "reason": "no recognised structure at that point"}
        if scope == "class":
            self.class_overrides[cls] = role      # NEUTRAL here also mutes an AUTO role
        elif scope == "region":
            _, comp = cv2.connectedComponents((seg.labels == cls).astype(np.uint8))
            blob = comp == comp[r, c]
            self.regions = [g for g in self.regions if not (g.mask & blob).any()]
            if role != Role.NEUTRAL:
                self.regions.append(TrackedRegion(cls, role, blob))
        else:
            raise ValueError("scope must be 'class' or 'region'")
        return {"ok": True, "class": self.names[cls], "role": role.name.lower(), "scope": scope}

    # --------------------------------------------------------------------- per frame
    def class_roles(self) -> dict[int, Role]:
        roles: dict[int, Role] = {}
        if self.mode == "auto":
            target, danger = self.profiles[self.procedure].for_step(self.step)
            for n in target:
                roles[self.names.index(n)] = Role.TARGET
            for n in danger:                       # danger wins if listed in both
                roles[self.names.index(n)] = Role.DANGER
        roles.update(self.class_overrides)
        return {k: v for k, v in roles.items() if v != Role.NEUTRAL}

    def assign(self, seg: SegResult, anchors: list[dict] | None = None,
               frame_rgb: np.ndarray | None = None) -> RoleOutput:
        """anchors: user marks from the simulator, re-projected every frame:
        [{"u": .., "v": .., "role": "target" | "danger"}] (normalised image coords).
        They work in both modes; in MANUAL mode they are the only source of roles."""
        prof = self.profiles[self.procedure]
        roles = self.class_roles()
        h, w = seg.hw
        role_map = np.zeros((h, w), np.uint8)
        marks = self._anchor_masks(seg, anchors or [], frame_rgb)

        target_ids = [c for c, r in roles.items() if r == Role.TARGET]
        danger_ids = [c for c, r in roles.items() if r == Role.DANGER]
        if target_ids:
            tgt = np.isin(seg.labels, target_ids) & (seg.confidence >= prof.target_threshold)
            role_map[tgt] = Role.TARGET
        self._track_regions(seg)
        for g in self.regions:
            if g.role == Role.TARGET:
                role_map[g.mask] = Role.TARGET
        for role, m in marks:
            if role == Role.TARGET:
                role_map[m] = Role.TARGET
        if danger_ids:
            p_danger = seg.probs[danger_ids].sum(0)
            role_map[p_danger >= prof.danger_threshold] = Role.DANGER
        for g in self.regions:
            if g.role == Role.DANGER:
                role_map[g.mask] = Role.DANGER
        for role, m in marks:                       # danger always wins
            if role == Role.DANGER:
                role_map[m] = Role.DANGER
        role_map[seg.labels == self.instrument] = Role.INSTRUMENT
        return RoleOutput(role_map, roles, self._regions(seg, role_map, prof.min_region_frac))

    def _anchor_masks(self, seg: SegResult, anchors: list[dict], frame_rgb) -> list:
        """User mark -> region. If the model recognises the tissue under the mark, take that
        whole connected structure; otherwise (unfamiliar simulator / unknown tissue) grow
        the region from the mark by colour similarity, so manual mode works on any scene."""
        h, w = seg.hw
        small, out = None, []
        for a in anchors:
            try:
                role = Role[str(a.get("role", "danger")).upper()]
                u, v = float(a["u"]), float(a["v"])
            except (KeyError, ValueError, TypeError):
                continue
            if role not in (Role.TARGET, Role.DANGER) or not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                continue                            # off screen / behind the camera this frame
            r, c = seg.to_px(u, v)
            cls = int(seg.labels[r, c])
            if cls == self.instrument:
                continue                            # a tool is covering the mark right now
            if cls != 0:
                _, comp = cv2.connectedComponents((seg.labels == cls).astype(np.uint8))
                mask = comp == comp[r, c]
                if mask.mean() > self.max_mark_frac:
                    # the model's region is huge (e.g. a simulator backdrop it calls one organ):
                    # keep only the part of it around the mark, not half the screen
                    radius = np.sqrt(self.max_mark_frac * h * w / np.pi) / max(h, w)
                    mask &= self._disc(h, w, r, c, radius)
            else:
                if small is None and frame_rgb is not None:
                    small = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_AREA)
                mask = self._grow(small, r, c) if small is not None else self._disc(h, w, r, c, 0.05)
            out.append((role, mask))
        return out

    @staticmethod
    def _grow(img: np.ndarray, r: int, c: int, tol: int = 14, max_frac: float = 0.25) -> np.ndarray:
        h, w = img.shape[:2]
        lab = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2LAB), (5, 5), 0)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
        cv2.floodFill(lab, mask, (int(c), int(r)), (0, 0, 0), (tol,) * 3, (tol,) * 3, flags)
        m = mask[1:-1, 1:-1]
        m = cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0   # recover the edge lost to the blur
        if m.mean() > max_frac:                     # leaked across the scene: use a disc
            m = RoleAssigner._disc(h, w, r, c, 0.05)
        return m

    @staticmethod
    def _disc(h: int, w: int, r: int, c: int, frac: float) -> np.ndarray:
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, (int(c), int(r)), max(2, int(frac * max(h, w))), 1, -1)
        return m.astype(bool)

    def _track_regions(self, seg: SegResult):
        """Follow each picked blob to the same-class component(s) it overlaps this frame.

        Components keep the region only if a good part of them overlapped it, so a
        neighbouring vessel of the same class is not swallowed. If nothing overlaps
        (fast motion), fall back to the single best match within a small search radius.
        """
        kept = []
        kernel = np.ones((15, 15), np.uint8)
        for g in self.regions:
            same = seg.labels == g.class_id
            n, comp = cv2.connectedComponents(same.astype(np.uint8))
            area = np.bincount(comp.ravel(), minlength=n)
            inter = np.bincount(comp[g.mask & same], minlength=n)
            inter[0] = 0
            ids = np.nonzero(inter >= 0.3 * np.maximum(area, 1))[0]
            if not len(ids) and inter.max() > 0:
                ids = [int(inter.argmax())]
            if not len(ids):
                grown = cv2.dilate(g.mask.astype(np.uint8), kernel).astype(bool)
                near = np.bincount(comp[grown & same], minlength=n)
                near[0] = 0
                ids = [int(near.argmax())] if near.max() > 0 else []
            if len(ids):
                g.mask, g.lost_frames = np.isin(comp, ids), 0
            else:
                g.lost_frames += 1          # occluded? keep the last mask for a while
            if g.lost_frames <= self.region_track_frames:
                kept.append(g)
        self.regions = kept

    def _regions(self, seg: SegResult, role_map: np.ndarray, min_frac: float) -> list[dict]:
        out = []
        h, w = role_map.shape
        for role in (Role.TARGET, Role.DANGER):
            n, comp, stats, cents = cv2.connectedComponentsWithStats(
                (role_map == role).astype(np.uint8))
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_frac * h * w:
                    continue
                blob = comp == i
                cls = np.bincount(seg.labels[blob], minlength=len(self.names)).argmax()
                # Anchor at the most interior pixel (centroids of C-shaped blobs can fall
                # outside them) so a ray cast from the camera actually hits this tissue.
                dist = cv2.distanceTransform(blob.astype(np.uint8), cv2.DIST_L2, 3)
                ry, rx = np.unravel_index(int(dist.argmax()), dist.shape)
                out.append({"role": role.name.lower(), "class": self.names[cls],
                            "anchor_uv": [(rx + 0.5) / w, (ry + 0.5) / h],
                            "area_frac": round(float(area) / (h * w), 4),
                            "mean_conf": round(float(seg.confidence[blob].mean()), 3)})
        return out


def colorize_roles(role_map: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 3), np.uint8)
    for r, c in ROLE_COLORS.items():
        lut[int(r)] = c
    return lut[role_map]
