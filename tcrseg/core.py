"""Trust-Component Router (TCR) inference utilities.

TCR-Seg is implemented as a click-time controller on top of a 2D interactive
segmentation backbone. The controller updates the backbone probability map with
three lightweight operations:

1. trust-guided context-prior modulation;
2. click-constrained threshold selection;
3. optional component-aware routing utilities from :mod:`tcrseg.component_routing`.

This file contains only framework-agnostic NumPy code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .clicks import Click, normalize_clicks


DEFAULT_THRESHOLD_CANDIDATES = (0.35, 0.40, 0.45, 0.49, 0.50, 0.55, 0.60, 0.65)


@dataclass
class TCRConfig:
    """Configuration used by the TCR inference controller."""

    prior_mode: str = "uncertain_trust_logit"
    prior_weight: float = 0.10
    prior_start_click: int = 1
    prior_click_margin: float = 0.05
    prior_trust_tau: float = 0.15
    prior_weight_schedule: tuple[tuple[int, float], ...] = field(default_factory=tuple)

    threshold_mode: str = "click_consistency_regularized"
    threshold_candidates: tuple[float, ...] = DEFAULT_THRESHOLD_CANDIDATES
    default_threshold: float = 0.49
    threshold_click_margin: float = 0.02
    threshold_default_penalty: float = 0.01

    eps: float = 1e-4


@dataclass
class TCRResult:
    """Output of one TCR update."""

    probs: np.ndarray
    mask: np.ndarray
    threshold: float
    trust: float
    diagnostics: dict[str, Any]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float32), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def as_hw_float(array: np.ndarray, name: str = "array") -> np.ndarray:
    """Return a squeezed ``H x W`` float32 map."""

    out = np.squeeze(np.asarray(array)).astype(np.float32)
    if out.ndim != 2:
        raise ValueError(f"Expected {name} to be a 2D map after squeeze, got shape {out.shape}")
    return out


def resize_prob_map(prob: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a probability map to ``shape=(height, width)`` using bilinear interpolation."""

    prob = as_hw_float(prob, "prob")
    if prob.shape == shape:
        return prob.astype(np.float32)

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only without OpenCV
        raise ImportError("opencv-python is required when resizing prior maps") from exc

    return cv2.resize(prob, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def scheduled_weight(default_weight: float, click_count: int, schedule: Sequence[Sequence[float]] | None) -> float:
    """Apply a piecewise-constant click schedule to a scalar weight."""

    if not schedule:
        return float(default_weight)

    current = float(default_weight)
    for start_click, weight in schedule:
        if int(click_count) >= int(start_click):
            current = float(weight)
        else:
            break
    return current


def prior_click_trust(
    prior_probs: np.ndarray,
    clicks: Iterable[Any],
    margin: float = 0.05,
    tau: float = 0.15,
) -> tuple[float, list[float]]:
    """Estimate whether the cached prior agrees with the click history.

    Positive clicks should fall on high prior probability and negative clicks
    should fall on low prior probability. Violations decay the trust by
    ``exp(-mean(violation) / tau)``.
    """

    prior = as_hw_float(prior_probs, "prior_probs")
    clicks = normalize_clicks(clicks)
    h, w = prior.shape
    pos_floor = 0.5 + float(margin)
    neg_ceil = 0.5 - float(margin)
    violations: list[float] = []

    for click in clicks:
        y, x = click.coords
        if y < 0 or x < 0 or y >= h or x >= w:
            continue
        value = float(prior[y, x])
        if click.is_positive:
            violations.append(max(0.0, pos_floor - value))
        else:
            violations.append(max(0.0, value - neg_ceil))

    if not violations:
        return 1.0, []

    tau = max(float(tau), 1e-6)
    trust = float(np.exp(-float(np.mean(violations)) / tau))
    return trust, violations


def trust_guided_logit_update(
    pred_probs: np.ndarray,
    prior_probs: np.ndarray | None,
    clicks: Iterable[Any],
    config: TCRConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fuse the backbone prediction with a cached context prior.

    The default ``uncertain_trust_logit`` mode follows the controller design:
    prior logits are injected mostly around uncertain pixels and are globally
    scaled by prior-click trust.
    """

    config = config or TCRConfig()
    clicks = normalize_clicks(clicks)
    probs = as_hw_float(pred_probs, "pred_probs")

    diagnostics: dict[str, Any] = {
        "applied": False,
        "mode": config.prior_mode,
        "trust": 1.0,
        "weight": 0.0,
        "mean_violation": 0.0,
    }

    weight = scheduled_weight(config.prior_weight, len(clicks), config.prior_weight_schedule)
    diagnostics["weight"] = float(weight)

    if (
        config.prior_mode == "none"
        or prior_probs is None
        or weight == 0.0
        or len(clicks) < int(config.prior_start_click)
    ):
        return probs.astype(np.float32), diagnostics

    prior = resize_prob_map(prior_probs, probs.shape)
    prior = np.clip(prior, config.eps, 1.0 - config.eps)
    probs = np.clip(probs, config.eps, 1.0 - config.eps)

    trust, violations = prior_click_trust(
        prior,
        clicks,
        margin=config.prior_click_margin,
        tau=config.prior_trust_tau,
    )
    diagnostics["trust"] = float(trust)
    diagnostics["mean_violation"] = float(np.mean(violations)) if violations else 0.0
    diagnostics["num_clicks"] = len(clicks)

    if trust <= 0.0:
        return probs.astype(np.float32), diagnostics

    pred_logits = logit(probs, config.eps)
    prior_logits = logit(prior, config.eps)
    scale = float(weight) * float(trust)

    if config.prior_mode == "trust_logit":
        logits = pred_logits + scale * prior_logits
    elif config.prior_mode == "uncertain_trust_logit":
        uncertainty = np.clip(1.0 - np.abs(probs - 0.5) * 2.0, 0.0, 1.0)
        logits = pred_logits + scale * uncertainty * prior_logits
        diagnostics["mean_uncertainty"] = float(np.mean(uncertainty))
    elif config.prior_mode == "trust_blend":
        alpha = min(max(scale, 0.0), 1.0)
        logits = (1.0 - alpha) * pred_logits + alpha * prior_logits
        diagnostics["alpha"] = float(alpha)
    else:
        raise ValueError(f"Unknown prior mode: {config.prior_mode}")

    diagnostics["applied"] = True
    return sigmoid(logits).astype(np.float32), diagnostics


def click_constrained_threshold(
    pred_probs: np.ndarray,
    clicks: Iterable[Any],
    default_threshold: float = 0.49,
    mode: str = "fixed",
    candidates: Sequence[float] | None = None,
    margin: float = 0.02,
    default_penalty: float = 0.01,
) -> tuple[float, dict[str, Any]]:
    """Select a threshold that is consistent with positive/negative clicks."""

    clicks = normalize_clicks(clicks)
    if mode == "fixed" or len(clicks) == 0:
        return float(default_threshold), {"mode": mode, "score": 0.0}

    if candidates is None:
        candidates = DEFAULT_THRESHOLD_CANDIDATES

    probs = as_hw_float(pred_probs, "pred_probs")
    best_score: float | None = None
    best_threshold = float(default_threshold)
    h, w = probs.shape

    for candidate in candidates:
        threshold = float(candidate)
        score = 0.0
        for click in clicks:
            y, x = click.coords
            if y < 0 or x < 0 or y >= h or x >= w:
                continue
            value = float(probs[y, x])
            if click.is_positive:
                score += max(0.0, threshold + margin - value)
            else:
                score += max(0.0, value - (threshold - margin))

        if mode == "click_consistency_regularized":
            score += float(default_penalty) * abs(threshold - float(default_threshold))
        elif mode != "click_consistency":
            raise ValueError(f"Unknown threshold mode: {mode}")

        if best_score is None or score < best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, {"mode": mode, "score": float(best_score or 0.0)}


def apply_tcr(
    pred_probs: np.ndarray,
    clicks: Iterable[Any],
    prior_probs: np.ndarray | None = None,
    config: TCRConfig | None = None,
) -> TCRResult:
    """Run the full TCR update for one interactive prediction step."""

    config = config or TCRConfig()
    updated_probs, prior_diag = trust_guided_logit_update(pred_probs, prior_probs, clicks, config)
    threshold, threshold_diag = click_constrained_threshold(
        updated_probs,
        clicks,
        default_threshold=config.default_threshold,
        mode=config.threshold_mode,
        candidates=config.threshold_candidates,
        margin=config.threshold_click_margin,
        default_penalty=config.threshold_default_penalty,
    )
    mask = updated_probs > threshold
    diagnostics = {"prior": prior_diag, "threshold": threshold_diag}
    return TCRResult(
        probs=updated_probs.astype(np.float32),
        mask=mask.astype(bool),
        threshold=float(threshold),
        trust=float(prior_diag.get("trust", 1.0)),
        diagnostics=diagnostics,
    )


def dice_iou(pred_mask: np.ndarray, target_mask: np.ndarray) -> tuple[float, float]:
    """Compute binary Dice and IoU for examples/tests."""

    pred = np.asarray(pred_mask).astype(bool)
    target = np.asarray(target_mask).astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    union = np.logical_or(pred, target).sum()
    dice = 1.0 if denom == 0 else 2.0 * float(intersection) / float(max(denom, 1))
    iou = 1.0 if union == 0 else float(intersection) / float(max(union, 1))
    return dice, iou
