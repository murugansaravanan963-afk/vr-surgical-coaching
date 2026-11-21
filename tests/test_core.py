"""Core correctness tests. Run: python -m pytest tests  (or: python tests/test_core.py)"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from surgseg.losses import PartialLabelLoss
from surgseg.metrics import PartialConfusion
from surgseg.runtime.engine import SegResult
from surgseg.runtime.roles import Role, RoleAssigner
from surgseg.runtime.safety import Intrinsics, SafetyConfig, SafetyMonitor, Tip
from surgseg.taxonomy import load_taxonomy

TAX = load_taxonomy()
C = TAX.num_classes
N = TAX.names.index


def test_full_labels_equal_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(2, C, 8, 8)
    target = torch.randint(0, C, (2, 8, 8))
    labeled = torch.ones(2, C, dtype=torch.bool)
    ours = PartialLabelLoss(TAX, dice_weight=0)(logits, target, labeled)["ce"]
    assert torch.allclose(ours, F.cross_entropy(logits, target), atol=1e-5)


def test_background_marginalises_unlabeled_classes():
    logits = torch.randn(1, C, 1, 1)
    labeled = torch.zeros(1, C, dtype=torch.bool)
    labeled[0, [N("liver"), N("instrument")]] = True
    loss = PartialLabelLoss(TAX, 0)(logits, torch.zeros(1, 1, 1, dtype=torch.long), labeled)["ce"]
    p = logits.softmax(1)[0, :, 0, 0]
    allowed = [c for c in range(C) if c not in (N("liver"), N("instrument"))]
    assert torch.allclose(loss, -p[allowed].sum().log(), atol=1e-5)


def test_group_label():
    logits = torch.randn(1, C, 1, 1)
    gid = next(g for g, (n, _) in TAX.groups.items() if n == "gi_tract")
    loss = PartialLabelLoss(TAX, 0)(logits, torch.full((1, 1, 1), gid), torch.ones(1, C).bool())["ce"]
    p = logits.softmax(1)[0, :, 0, 0]
    assert torch.allclose(loss, -p[[N("stomach"), N("small_intestine"), N("colon")]].sum().log(),
                          atol=1e-5)


def test_ignore_gives_finite_gradients():
    logits = torch.randn(2, C, 6, 6, requires_grad=True)
    target = torch.full((2, 6, 6), 255)
    target[0, 0, 0] = N("liver")
    labeled = torch.zeros(2, C, dtype=torch.bool)
    labeled[0, N("liver")] = True
    PartialLabelLoss(TAX, 0.5)(logits, target, labeled)["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_metric_ignores_unlabeled_predictions():
    conf = PartialConfusion(C, 1)
    target = torch.tensor([[[N("liver"), 0]]])
    pred = torch.tensor([[[N("liver"), N("stomach")]]])   # stomach not labeled -> counts as bg
    labeled = torch.zeros(1, C, dtype=torch.bool)
    labeled[0, N("liver")] = True
    conf.update(pred, target, labeled, torch.tensor([0]))
    assert conf.summary(TAX.names, ["d"])["d"]["mIoU"] == 1.0


def _scene():
    """Gallbladder on the left, a duct in the middle, instrument on the right."""
    h, w = 64, 64
    labels = np.zeros((h, w), np.uint8)
    labels[:, :20] = N("gallbladder")
    labels[:, 30:34] = N("duct")
    labels[:, 50:] = N("instrument")
    probs = np.full((C, h, w), 0.01, np.float32)
    for c in np.unique(labels):
        probs[c][labels == c] = 0.9
    probs /= probs.sum(0, keepdims=True)
    return SegResult(probs, labels, probs.max(0), (h, w))


def test_auto_mode_profile():
    seg = _scene()
    ra = RoleAssigner(TAX.names, mode="auto", procedure="cholecystectomy")
    rm = ra.assign(seg).role_map
    assert rm[10, 5] == Role.TARGET and rm[10, 31] == Role.DANGER and rm[10, 55] == Role.INSTRUMENT
    # steps change the knowledge base: in Calot dissection the gallbladder is not the target
    ra.set_mode("auto", "cholecystectomy", step="calot_dissection")
    assert ra.assign(seg).role_map[10, 5] == Role.NEUTRAL


def test_manual_mode_selection():
    seg = _scene()
    ra = RoleAssigner(TAX.names, mode="manual", procedure="default")
    assert (ra.assign(seg).role_map == Role.NEUTRAL).sum() == (seg.labels != N("instrument")).sum()
    assert ra.select(seg, u=5 / 64, v=0.5, role="danger", scope="class")["class"] == "gallbladder"
    assert ra.select(seg, u=55 / 64, v=0.5, role="target")["ok"] is False   # instrument
    rm = ra.assign(seg).role_map
    assert rm[10, 5] == Role.DANGER and rm[10, 31] == Role.NEUTRAL


def test_region_scope_tracks_one_blob():
    seg = _scene()
    seg.labels[:, 40:44] = N("duct")                       # a second duct blob
    ra = RoleAssigner(TAX.names, mode="manual")
    ra.select(seg, u=31.5 / 64, v=0.5, role="danger", scope="region")
    rm = ra.assign(seg).role_map
    assert rm[10, 31] == Role.DANGER and rm[10, 41] == Role.NEUTRAL
    seg.labels[:, 30:34], seg.labels[:, 31:35] = 0, N("duct")   # blob moves 1px
    assert ra.assign(seg).role_map[10, 34] == Role.DANGER


def test_safety_levels_and_hysteresis():
    rm = np.zeros((100, 100), np.uint8)
    rm[:, 50:] = Role.DANGER
    labels = np.full((100, 100), N("duct"), np.uint8)
    mon = SafetyMonitor(SafetyConfig(warn_px=20, contact_px=3, clear_after_frames=2))
    lvl = lambda u: mon.update(rm, labels, TAX.names, [Tip("t", u, 0.5)])[0]["level"]
    assert lvl(0.10) == "ok"
    assert lvl(0.40) == "warning"
    assert lvl(0.49) == "violation"
    assert lvl(0.10) == "violation"         # held for one calm frame
    assert lvl(0.10) == "ok"                # cleared after two
    stale = mon.update(rm, labels, TAX.names, [Tip("t", 0.1, 0.5)], frame_age_ms=500)
    assert stale[0]["level"] == "stale"


def test_safety_3d_distance():
    rm = np.zeros((100, 100), np.uint8)
    rm[:, 60:] = Role.DANGER
    depth = np.full((100, 100), 0.10, np.float32)            # flat tissue 10 cm away
    K = Intrinsics(fx=100, fy=100, cx=50, cy=50, width=100, height=100)
    # 1 px = 1 mm at 10 cm. Tip hovers 4 mm above the tissue, 1.5 mm short of the danger
    # edge (first danger pixel centre is x = 10.5 mm) -> sqrt(1.5^2 + 4^2) = 4.27 mm.
    tip = Tip("t", u=0.59, v=0.5, xyz=(0.009, 0.0005, 0.096))
    rep = SafetyMonitor().update(rm, np.zeros_like(rm), TAX.names, [tip], 0, depth, K)[0]
    assert abs(rep["dist_mm"] - 4.27) < 0.05, rep
    assert rep["level"] == "warning"                         # < 10 mm warn, > 2 mm contact


def test_partial_label_does_not_extend_labeled_set():
    # Endoscapes: "cystic plate" -> partial:connective_tissue. The pixel is supervised as
    # connective tissue, but Endoscapes background must still PERMIT connective tissue.
    assert N("connective_tissue") not in TAX.labeled_set("endoscapes")
    v = TAX.resolve("partial:connective_tissue")
    m = TAX.allowed_matrix(TAX.labeled_set("endoscapes"))
    assert m[v].sum() == 1 and m[v, N("connective_tissue")]
    assert m[0, N("connective_tissue")] and not m[0, N("artery")]


def test_label_completion_respects_ground_truth():
    from surgseg.pseudo_label import complete
    labeled = [N("instrument"), N("kidney")]              # EndoVis18-like sample
    allowed = TAX.allowed_matrix(labeled)
    gt = np.array([[0, 0, N("kidney"), 255]], np.uint8)
    probs = np.full((C, 1, 4), 0.001, np.float32)
    probs[N("liver"), 0, 0] = 0.95     # bg pixel, confident liver   -> filled
    probs[N("kidney"), 0, 1] = 0.95    # bg pixel, but kidney is labeled -> contradiction, kept
    probs[N("liver"), 0, 2] = 0.95     # GT already says kidney     -> kept
    probs[N("liver"), 0, 3] = 0.95     # ignore stays ignore
    out = complete(gt, probs, allowed, tau=0.9)
    assert out.tolist() == [[N("liver"), 0, N("kidney"), 255]]


def test_sampler_boosts_rare_classes_within_dataset():
    from surgseg.data import InfiniteWeightedSampler
    recs = [{"dataset": "a", "present": [N("duct")]}] + [{"dataset": "a", "present": []}] * 9 \
         + [{"dataset": "b", "present": []}] * 10
    s = InfiniteWeightedSampler(recs, {"a": 1.0, "b": 1.0}, 0, 0, boost_classes=[N("duct")],
                                boost_factor=9.0)
    w = s.weights / s.weights.sum()
    assert abs(float(w[:10].sum()) - 0.5) < 1e-9          # dataset shares unchanged
    assert abs(float(w[0]) - 0.25) < 1e-9                 # duct frame: 9/18 of dataset a


def test_class_weights_upweight_danger_pixels():
    torch.manual_seed(0)
    logits = torch.randn(1, C, 1, 2)
    target = torch.tensor([[[N("artery"), N("liver")]]])
    labeled = torch.ones(1, C, dtype=torch.bool)
    nll = -F.log_softmax(logits, 1)[0, :, 0]                   # (C, 2)
    a, l = nll[N("artery"), 0], nll[N("liver"), 1]
    got = PartialLabelLoss(TAX, 0, {"artery": 3.0})(logits, target, labeled)["ce"]
    assert torch.allclose(got, (3 * a + l) / 4, atol=1e-5)


def test_val_picker_covers_rare_classes_without_starving_train():
    from surgseg.tools.refine_index import pick_val_groups
    d, v = N("duct"), N("vein")
    counts = {"v1": {d: 100}, "v2": {d: 100, v: 50}, "v3": {v: 50}, "v4": {}, "v5": {d: 100},
              "only": {N("blood"): 10}}
    val = pick_val_groups(counts, sorted(counts), [d, v, N("blood")], n=2)
    assert "only" not in val                        # sole blood video must stay in train
    covered = {c for g in val for c in counts[g]}
    assert d in covered and v in covered
    for c in (d, v):                                # at most half of each class in val
        assert sum(counts[g].get(c, 0) for g in val) <= 0.5 * sum(x.get(c, 0) for x in counts.values())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
