"""Component-aware click routing and mask update utilities.

The next-click selectors in this module use ground truth and are intended for
benchmark evaluation, where automatic click simulation is standard. They should
not be used as a deployed user-click policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .clicks import Click, as_click


@dataclass
class RoutingConfig:
    strategy: str = "distance_then_component_area"
    switch_after: int = 9
    click_topk: int = 128
    uncertainty_weight: float = 0.0
    boundary_weight: float = 0.0


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only without OpenCV
        raise ImportError("opencv-python is required for component routing") from exc
    return cv2


def distance_transform(mask: np.ndarray, padding: bool = True) -> np.ndarray:
    cv2 = _cv2()
    mask = np.asarray(mask).astype(bool)
    if padding:
        mask = np.pad(mask, ((1, 1), (1, 1)), "constant")
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 0)
    if padding:
        dist = dist[1:-1, 1:-1]
    return dist.astype(np.float32)


def progressive_merge(candidate_mask: np.ndarray, previous_mask: np.ndarray, click: Any) -> np.ndarray:
    """Merge only the connected changed component touched by the latest click."""

    cv2 = _cv2()
    click = as_click(click)
    candidate = np.asarray(candidate_mask).astype(bool)
    previous = np.asarray(previous_mask).astype(bool)
    if candidate.shape != previous.shape:
        raise ValueError(f"Mask shape mismatch: {candidate.shape} vs {previous.shape}")

    y, x = click.coords
    if y < 0 or x < 0 or y >= candidate.shape[0] or x >= candidate.shape[1]:
        return candidate

    changed = np.logical_xor(previous, candidate)
    num_labels, labels = cv2.connectedComponents(changed.astype(np.uint8))
    if num_labels <= 1:
        return candidate

    label = int(labels[y, x])
    if label == 0:
        return candidate

    selected_component = labels == label
    if previous[y, x]:
        return np.logical_and(previous, np.logical_not(selected_component))
    return np.logical_or(previous, selected_component)


def prompt_component_projection(pred_mask: np.ndarray, clicks: list[Any], mode: str = "none") -> np.ndarray:
    """Keep or suppress connected components according to clicked components."""

    if mode == "none" or len(clicks) == 0:
        return np.asarray(pred_mask).astype(bool)

    cv2 = _cv2()
    mask = np.asarray(pred_mask).astype(bool)
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if num_labels <= 1:
        return mask

    pos_labels: set[int] = set()
    neg_labels: set[int] = set()
    h, w = mask.shape
    for raw_click in clicks:
        click = as_click(raw_click)
        y, x = click.coords
        if y < 0 or x < 0 or y >= h or x >= w:
            continue
        label = int(labels[y, x])
        if label == 0:
            continue
        if click.is_positive:
            pos_labels.add(label)
        else:
            neg_labels.add(label)

    if mode == "largest_component":
        areas = np.bincount(labels.reshape(-1))
        areas[0] = 0
        keep_label = int(np.argmax(areas))
        return labels == keep_label if keep_label > 0 else mask
    if mode == "positive_components":
        keep_labels = pos_labels
    elif mode == "positive_negative_components":
        keep_labels = pos_labels - neg_labels
    else:
        raise ValueError(f"Unknown component projection mode: {mode}")

    if keep_labels:
        return np.isin(labels, list(keep_labels))
    if mode == "positive_negative_components" and neg_labels:
        return np.logical_and(mask, np.logical_not(np.isin(labels, list(neg_labels))))
    return mask


def select_next_click(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    pred_probs: np.ndarray | None = None,
    clicked_map: np.ndarray | None = None,
    click_count: int = 0,
    config: RoutingConfig | None = None,
) -> Click:
    """Select the next simulated click for evaluation.

    ``gt_mask`` is required because benchmark click simulation places clicks in
    false-negative or false-positive error regions.
    """

    config = config or RoutingConfig()
    strategy = config.strategy
    if strategy in {"distance_then_component_area", "distance_then_component_uncertainty"}:
        if click_count < config.switch_after:
            strategy = "distance"
        elif strategy == "distance_then_component_area":
            strategy = "component_area"
        else:
            strategy = "component_uncertainty"

    gt = np.asarray(gt_mask).astype(bool)
    pred = np.asarray(pred_mask).astype(bool)
    if gt.shape != pred.shape:
        raise ValueError(f"Mask shape mismatch: {gt.shape} vs {pred.shape}")

    available = np.ones_like(gt, dtype=bool) if clicked_map is None else np.asarray(clicked_map).astype(bool)
    fn_mask = np.logical_and(gt, np.logical_not(pred))
    fp_mask = np.logical_and(np.logical_not(gt), pred)

    if strategy == "component_area":
        fn_score, fn_coords = _largest_component_center(fn_mask, available)
        fp_score, fp_coords = _largest_component_center(fp_mask, available)
        if fn_score > 0 or fp_score > 0:
            positive = fn_score > fp_score
            coords = fn_coords if positive else fp_coords
            return Click(int(coords[0]), int(coords[1]), positive=positive, index=click_count)

    if strategy in {"component_uncertainty", "distance_then_component_uncertainty"}:
        uncertainty = _uncertainty_map(pred_probs, gt.shape)
        fn_score, fn_coords = _component_guided_center(fn_mask, available, uncertainty, config.uncertainty_weight)
        fp_score, fp_coords = _component_guided_center(fp_mask, available, uncertainty, config.uncertainty_weight)
        if fn_score > 0 or fp_score > 0:
            positive = fn_score > fp_score
            coords = fn_coords if positive else fp_coords
            return Click(int(coords[0]), int(coords[1]), positive=positive, index=click_count)

    return _distance_click(fn_mask, fp_mask, available, click_count)


def _distance_click(fn_mask: np.ndarray, fp_mask: np.ndarray, available: np.ndarray, click_count: int) -> Click:
    fn_dist = distance_transform(fn_mask) * available
    fp_dist = distance_transform(fp_mask) * available
    fn_max = float(np.max(fn_dist))
    fp_max = float(np.max(fp_dist))
    positive = fn_max > fp_max
    score = fn_dist if positive else fp_dist
    coords_y, coords_x = np.where(score == np.max(score))
    if len(coords_y) == 0:
        coords_y, coords_x = np.where(available)
    return Click(int(coords_y[0]), int(coords_x[0]), positive=positive, index=click_count)


def _largest_component_center(mask: np.ndarray, available: np.ndarray) -> tuple[float, tuple[int, int]]:
    cv2 = _cv2()
    dt = distance_transform(mask) * available
    num_labels, labels = cv2.connectedComponents(np.asarray(mask).astype(np.uint8))
    best_area = 0.0
    best_dist = 0.0
    best_coords = (0, 0)

    for component_id in range(1, num_labels):
        component = labels == component_id
        component_dt = dt * component
        max_dist = float(np.max(component_dt))
        if max_dist <= 0:
            continue
        area = float(component.sum())
        if area > best_area or (area == best_area and max_dist > best_dist):
            coords_y, coords_x = np.where(component_dt == max_dist)
            best_area = area
            best_dist = max_dist
            best_coords = (int(coords_y[0]), int(coords_x[0]))

    return best_area, best_coords


def _component_guided_center(
    mask: np.ndarray,
    available: np.ndarray,
    uncertainty: np.ndarray | None,
    uncertainty_weight: float,
) -> tuple[float, tuple[int, int]]:
    cv2 = _cv2()
    dt = distance_transform(mask) * available
    num_labels, labels = cv2.connectedComponents(np.asarray(mask).astype(np.uint8))
    best_score = 0.0
    best_coords = (0, 0)

    for component_id in range(1, num_labels):
        component = labels == component_id
        component_dt = dt * component
        max_dist = float(np.max(component_dt))
        if max_dist <= 0:
            continue
        area = float(component.sum())
        mean_uncertainty = float(np.mean(uncertainty[component])) if uncertainty is not None else 0.0
        score = area * (1.0 + float(uncertainty_weight) * mean_uncertainty)
        guided_map = component_dt.copy()
        if uncertainty is not None:
            guided_map += float(uncertainty_weight) * max_dist * uncertainty * component
        coords_y, coords_x = np.where(guided_map == np.max(guided_map))
        if score > best_score:
            best_score = score
            best_coords = (int(coords_y[0]), int(coords_x[0]))

    return best_score, best_coords


def _uncertainty_map(pred_probs: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if pred_probs is None:
        return None
    probs = np.squeeze(np.asarray(pred_probs)).astype(np.float32)
    if probs.shape != shape:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ImportError("opencv-python is required when resizing probability maps") from exc
        probs = cv2.resize(probs, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    probs = np.clip(probs, 0.0, 1.0)
    return np.clip(1.0 - np.abs(probs - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
