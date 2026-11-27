"""Partial-label losses for training one model on datasets with different label sets.

Each mask value permits a SET of unified classes (see Taxonomy.allowed_matrix):
  * a class id permits only that class           -> ordinary cross-entropy
  * a group id ("GI tract") permits its members  -> -log(sum of member probs)
  * 0 permits background + every class the sample did not label
This is the "marginal loss" of Shi et al., MedIA 2021, generalised to groups.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from surgseg.taxonomy import IGNORE, PARTIAL_BASE, Taxonomy


class PartialLabelLoss(torch.nn.Module):
    def __init__(self, tax: Taxonomy, dice_weight: float = 0.5,
                 class_weights: dict[str, float] | None = None, lovasz_weight: float = 0.0,
                 lovasz_stride: int = 2):
        super().__init__()
        base = torch.from_numpy(tax.allowed_matrix(labeled=[]))   # row 0 is per-sample
        self.register_buffer("base", base, persistent=False)
        self.dice_weight = dice_weight
        # Lovasz-softmax optimises IoU directly (Berman et al., CVPR 2018). It helps small,
        # thin classes most, because IoU punishes every missed pixel of a 2-pixel-wide duct.
        self.lovasz_weight = lovasz_weight
        self.lovasz_stride = lovasz_stride   # subsample pixels for speed (sorting is costly)
        # Per-pixel CE weight by mask value: a pixel labelled "artery" (or partial:artery)
        # counts e.g. 3x. Thin danger structures are a tiny share of pixels otherwise.
        w = torch.ones(256)
        for name, cw in (class_weights or {}).items():
            c = tax.class_id(name)
            w[c] = cw
            w[PARTIAL_BASE + c] = cw
        self.register_buffer("value_weight", w, persistent=False)

    def allowed(self, target: torch.Tensor, labeled: torch.Tensor) -> torch.Tensor:
        """(B,H,W) target + (B,C) labeled -> (B,H,W,C) bool of permitted classes."""
        allowed = self.base[target]
        bg_row = ~labeled
        bg_row[:, 0] = True
        return torch.where((target == 0)[..., None], bg_row[:, None, None, :], allowed)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, labeled: torch.Tensor):
        logits = logits.float()
        valid = target != IGNORE
        allowed = self.allowed(target, labeled)                         # B,H,W,C
        logp = F.log_softmax(logits, dim=1).permute(0, 2, 3, 1)         # B,H,W,C
        # Ignored pixels permit everything so their logsumexp stays finite: an all -inf
        # row would back-propagate NaN even though it is masked out of the mean.
        permit = allowed | ~valid[..., None]
        marg = torch.logsumexp(logp.masked_fill(~permit, float("-inf")), dim=-1)
        if valid.any():
            w = self.value_weight[target][valid]
            ce = -(marg[valid] * w).sum() / w.sum()
        else:
            ce = logits.sum() * 0
        out = {"ce": ce}
        loss = ce
        if self.dice_weight > 0:
            dice = self.partial_dice(logp.exp(), allowed, valid, labeled)
            out["dice"] = dice
            loss = loss + self.dice_weight * dice
        if self.lovasz_weight > 0:
            s = self.lovasz_stride
            lov = self.partial_lovasz(logp.exp()[:, ::s, ::s], allowed[:, ::s, ::s],
                                      valid[:, ::s, ::s], labeled)
            out["lovasz"] = lov
            loss = loss + self.lovasz_weight * lov
        out["loss"] = loss
        return out

    @staticmethod
    def partial_lovasz(prob, allowed, valid, labeled):
        """Lovasz-softmax over pixels whose membership in class c is known (same rule as Dice).

        For each class present in the batch: errors = |fg - p_c| on the known pixels, sorted
        descending, weighted by the gradient of the Lovasz extension of the Jaccard loss.
        """
        exact = allowed.sum(-1, keepdim=True) == 1
        pos = allowed & exact
        known = (pos | ~allowed) & valid[..., None] & labeled[:, None, None, :]
        C = prob.shape[-1]
        losses = []
        for c in range(1, C):                       # background is not a target structure
            k = known[..., c]
            fg = pos[..., c][k].float()
            if fg.sum() == 0:
                continue
            err = (fg - prob[..., c][k]).abs()
            err_sorted, order = torch.sort(err, descending=True)
            fg_sorted = fg[order]
            gts = fg_sorted.sum()
            inter = gts - fg_sorted.cumsum(0)
            union = gts + (1 - fg_sorted).cumsum(0)
            jac = 1.0 - inter / union
            grad = torch.cat([jac[:1], jac[1:] - jac[:-1]])
            losses.append(torch.dot(err_sorted, grad))
        if not losses:
            return prob.sum() * 0
        return torch.stack(losses).mean()

    @staticmethod
    def partial_dice(prob, allowed, valid, labeled):
        """Batch soft-Dice over pixels whose membership in class c is actually known."""
        exact = allowed.sum(-1, keepdim=True) == 1
        pos = allowed & exact                                   # definitely class c
        known = (pos | ~allowed) & valid[..., None] & labeled[:, None, None, :]
        pos = (pos & known).float()
        known = known.float()
        inter = (prob * pos).sum((0, 1, 2))
        denom = (prob * known).sum((0, 1, 2)) + pos.sum((0, 1, 2))
        present = pos.sum((0, 1, 2)) > 0
        if not present.any():
            return prob.sum() * 0
        dice = 1 - (2 * inter + 1) / (denom + 1)
        return dice[present].mean()


def distillation_loss(student_logits, teacher_probs, valid, T: float = 2.0):
    """KL(teacher || student) at temperature T; teacher_probs are already softened."""
    s = F.log_softmax(student_logits.float() / T, dim=1)
    t = teacher_probs.float()
    kl = (t * (t.clamp_min(1e-8).log() - s)).sum(1)
    return kl[valid].mean() * T * T if valid.any() else kl.sum() * 0
