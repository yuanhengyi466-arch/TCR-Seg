"""Run a dependency-light synthetic TCR-Seg demo.

This example does not require external datasets or checkpoints. It creates a
toy image, a noisy cached prior, several clicks, and a simple click-conditioned
probability predictor to verify that the TCR controller runs end to end.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcrseg import Click, TCRConfig, TCRSegmenter, dice_iou


def make_ellipse(shape: tuple[int, int], center: tuple[int, int], radii: tuple[int, int]) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    cy, cx = center
    ry, rx = radii
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def blur_like(prob: np.ndarray, steps: int = 6) -> np.ndarray:
    out = prob.astype(np.float32)
    for _ in range(steps):
        out = (
            out
            + np.roll(out, 1, 0)
            + np.roll(out, -1, 0)
            + np.roll(out, 1, 1)
            + np.roll(out, -1, 1)
        ) / 5.0
    return np.clip(out, 0.0, 1.0)


def gaussian(shape: tuple[int, int], y: int, x: int, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma**2)).astype(np.float32)


def main() -> None:
    shape = (160, 160)
    gt = make_ellipse(shape, center=(82, 86), radii=(38, 48))
    image = np.dstack([blur_like(gt, 10) * 180 + 35] * 3).astype(np.uint8)
    prior = np.clip(blur_like(gt.astype(np.float32), 12) * 0.86 + 0.06, 0.02, 0.98)

    clicks = [
        Click(78, 82, True),
        Click(80, 132, True),
        Click(48, 38, False),
    ]

    def toy_predictor(image: np.ndarray, clicks: list[Click], prev_mask=None) -> np.ndarray:
        logits = -1.8 + 2.2 * blur_like(gt.astype(np.float32), 4)
        for click in clicks:
            pulse = gaussian(shape, click.y, click.x, sigma=19.0)
            logits += (1.15 if click.positive else -1.35) * pulse
        return 1.0 / (1.0 + np.exp(-logits))

    segmenter = TCRSegmenter(
        predict_fn=toy_predictor,
        config=TCRConfig(),
    )

    base_probs = toy_predictor(image=image, clicks=clicks)
    base_mask = base_probs > 0.49
    result = segmenter.predict(image=image, clicks=clicks, prior_probs=prior)

    base_dice, base_iou = dice_iou(base_mask, gt)
    tcr_dice, tcr_iou = dice_iou(result.mask, gt)
    print(f"Base Dice/IoU: {base_dice:.4f}/{base_iou:.4f}")
    print(f"TCR  Dice/IoU: {tcr_dice:.4f}/{tcr_iou:.4f}")
    print(f"Threshold: {result.threshold:.2f}; prior trust: {result.trust:.3f}")

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_numpy.png"

    panels = [
        ("Image", image),
        ("Cached prior", prior),
        ("Backbone mask", base_mask),
        ("TCR mask", result.mask),
        ("GT", gt),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 2.2), dpi=180)
    for ax, (title, panel) in zip(axes, panels):
        if panel.ndim == 2:
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(panel)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
