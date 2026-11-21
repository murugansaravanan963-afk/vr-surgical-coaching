"""Region-entry warnings for the instrument tips.

The simulator KNOWS where each tool tip is, so we do not segment the tool to find it:
the engine projects each tip into the endoscope image and sends (u, v), optionally its
camera-frame 3D position plus the depth buffer. Segmentation only answers "what tissue
is around that point". Note the tip pixel itself is usually labelled "instrument"
(the tool occludes the tissue under it), which is why we measure DISTANCE to the
nearest danger pixel rather than testing the pixel under the tip.

Levels: OK < WARNING (within warn margin) < VIOLATION (within contact distance).
STALE is reported instead when the frame is too old to trust (proposal risk #2).
Levels rise immediately but only fall after `clear_after_frames` calmer frames, so the
alert doesn't flicker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import cv2
import numpy as np

from surgseg.runtime.roles import Role


class Level(IntEnum):
    OK = 0
    WARNING = 1
    VIOLATION = 2
    STALE = 3


@dataclass
class SafetyConfig:
    warn_px: float = 40.0          # 2D fallback margins, at model resolution
    contact_px: float = 6.0
    warn_mm: float = 10.0          # used when depth + intrinsics are available
    contact_mm: float = 2.0
    max_frame_age_ms: float = 150.0
    clear_after_frames: int = 3


@dataclass
class Tip:
    id: str
    u: float                        # normalised image coords [0, 1]
    v: float
    xyz: tuple[float, float, float] | None = None   # camera frame, metres (optional)


@dataclass
class Intrinsics:
    """Pinhole intrinsics of the frame sent to the server, in that frame's pixels."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class _TipState:
    level: Level = Level.OK
    calm: int = 0


@dataclass
class SafetyMonitor:
    cfg: SafetyConfig = field(default_factory=SafetyConfig)
    _state: dict = field(default_factory=dict)

    def update(self, role_map: np.ndarray, class_map: np.ndarray, class_names: list[str],
               tips: list[Tip], frame_age_ms: float = 0.0,
               depth: np.ndarray | None = None, K: Intrinsics | None = None) -> list[dict]:
        h, w = role_map.shape
        danger = role_map == Role.DANGER
        target = role_map == Role.TARGET
        d_danger = cv2.distanceTransform((~danger).astype(np.uint8), cv2.DIST_L2, 5)
        d_target = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 5)
        out = []
        for tip in tips:
            r = int(np.clip(tip.v * h, 0, h - 1))
            c = int(np.clip(tip.u * w, 0, w - 1))
            rep = {"id": tip.id, "dist_px": None, "dist_mm": None, "nearest_danger": None,
                   "on_target": bool(d_target[r, c] <= self.cfg.contact_px)}
            raw = Level.OK
            if danger.any():
                dpx = float(d_danger[r, c])
                rep["dist_px"] = round(dpx, 1)
                if dpx <= self.cfg.warn_px * 3:
                    rr, cc = self._nearest(danger, r, c, int(self.cfg.warn_px * 3) + 1)
                    if rr is not None:
                        rep["nearest_danger"] = class_names[int(class_map[rr, cc])]
                dmm = None
                if depth is not None and K is not None and tip.xyz is not None:
                    dmm = self._dist_mm_3d(danger, depth, K, tip, (h, w))
                if dmm is not None:
                    rep["dist_mm"] = round(dmm, 1)
                    raw = (Level.VIOLATION if dmm <= self.cfg.contact_mm else
                           Level.WARNING if dmm <= self.cfg.warn_mm else Level.OK)
                else:
                    raw = (Level.VIOLATION if dpx <= self.cfg.contact_px else
                           Level.WARNING if dpx <= self.cfg.warn_px else Level.OK)
            level = self._hysteresis(tip.id, raw)
            if frame_age_ms > self.cfg.max_frame_age_ms:
                level = Level.STALE
            rep["level"] = level.name.lower()
            out.append(rep)
        return out

    def _hysteresis(self, tip_id: str, raw: Level) -> Level:
        st = self._state.setdefault(tip_id, _TipState())
        if raw >= st.level:
            st.level, st.calm = raw, 0
        else:
            st.calm += 1
            if st.calm >= self.cfg.clear_after_frames:
                st.level, st.calm = raw, 0
        return st.level

    @staticmethod
    def _nearest(mask, r, c, rad):
        h, w = mask.shape
        r0, r1, c0, c1 = max(0, r - rad), min(h, r + rad + 1), max(0, c - rad), min(w, c + rad + 1)
        ys, xs = np.nonzero(mask[r0:r1, c0:c1])
        if len(ys) == 0:
            return None, None
        i = int(np.argmin((ys + r0 - r) ** 2 + (xs + c0 - c) ** 2))
        return ys[i] + r0, xs[i] + c0

    def _dist_mm_3d(self, danger, depth, K: Intrinsics, tip: Tip, hw):
        """Min 3D distance from the tip to danger-surface points near it (depth in metres)."""
        h, w = hw
        tx, ty, tz = tip.xyz
        if tz <= 0:
            return None
        # Search window: the warn radius projected into the image at the tip's depth.
        rad_px = self.cfg.warn_mm * 1e-3 * K.fx / tz * (w / K.width) * 2 + 3
        rr, cc = int(tip.v * h), int(tip.u * w)
        rad = int(min(rad_px, max(h, w)))
        r0, r1 = max(0, rr - rad), min(h, rr + rad + 1)
        c0, c1 = max(0, cc - rad), min(w, cc + rad + 1)
        ys, xs = np.nonzero(danger[r0:r1, c0:c1])
        if len(ys) == 0:
            return float("inf")
        ys, xs = ys + r0, xs + c0
        # Map model-resolution pixels to depth-buffer pixels and back-project.
        dh, dw = depth.shape
        py = ((ys + 0.5) * dh / h).astype(int).clip(0, dh - 1)
        px = ((xs + 0.5) * dw / w).astype(int).clip(0, dw - 1)
        z = depth[py, px]
        ok = np.isfinite(z) & (z > 0)
        if not ok.any():
            return None
        u_img = (xs[ok] + 0.5) * K.width / w
        v_img = (ys[ok] + 0.5) * K.height / h
        X = (u_img - K.cx) * z[ok] / K.fx
        Y = (v_img - K.cy) * z[ok] / K.fy
        d = np.sqrt((X - tx) ** 2 + (Y - ty) ** 2 + (z[ok] - tz) ** 2)
        return float(d.min() * 1000.0)
