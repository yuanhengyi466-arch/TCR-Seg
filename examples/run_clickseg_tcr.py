"""Example adapter for running TCR-Seg with a ClickSEG/FocalClick checkpoint.

This script intentionally does not vendor ClickSEG. Pass the path to an
installed ClickSEG repository and a checkpoint file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcrseg import Click, TCRConfig, apply_tcr


class StaticClicker:
    """Minimal ClickSEG-compatible click container."""

    def __init__(self, clickseg_click_cls, clicks: list[Click]):
        self.clicks_list = [
            clickseg_click_cls(is_positive=click.positive, coords=(click.y, click.x), indx=idx)
            for idx, click in enumerate(clicks)
        ]

    def get_clicks(self):
        return self.clicks_list


def load_clickseg_predictor(args: argparse.Namespace):
    sys.path.insert(0, str(args.clickseg_repo.resolve()))
    import torch
    from isegm.inference import utils as isegm_utils
    from isegm.inference.predictors import get_predictor

    device = torch.device(args.device)
    model = isegm_utils.load_is_model(args.checkpoint, device)
    predictor = get_predictor(
        model,
        args.mode,
        device,
        infer_size=args.infer_size,
        prob_thresh=args.threshold,
        predictor_params={"optimize_after_n_clicks": 1},
        focus_crop_r=args.focus_crop_r,
        focus_crop_mode=args.focus_crop_mode,
        late_coarse_mode=args.late_coarse_mode,
        late_coarse_switch_after=args.late_coarse_switch_after,
        late_coarse_blend=args.late_coarse_blend,
        zoom_in_params={
            "target_size": args.target_size,
            "expansion_ratio": args.expansion_ratio,
            "skip_clicks": -1,
        },
    )
    from isegm.inference.clicker import Click as ClickSEGClick

    return predictor, ClickSEGClick


def parse_clicks(raw: str) -> list[Click]:
    data = json.loads(raw)
    return [Click(int(item["y"]), int(item["x"]), bool(item.get("positive", True))) for item in data]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clickseg-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prior-npy", type=Path, default=None)
    parser.add_argument("--clicks-json", required=True, help='Example: "[{\"y\":80,\"x\":90,\"positive\":true}]"')
    parser.add_argument("--output-mask", type=Path, default=Path("tcr_mask.png"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="FocalClick")
    parser.add_argument("--infer-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.49)
    parser.add_argument("--focus-crop-r", type=float, default=1.4)
    parser.add_argument("--focus-crop-mode", default="click")
    parser.add_argument("--late-coarse-mode", default="blend")
    parser.add_argument("--late-coarse-switch-after", type=int, default=10)
    parser.add_argument("--late-coarse-blend", type=float, default=0.5)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--expansion-ratio", type=float, default=1.4)
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    clicks = parse_clicks(args.clicks_json)
    prior = np.load(args.prior_npy) if args.prior_npy is not None else None

    predictor, clickseg_click_cls = load_clickseg_predictor(args)
    predictor.set_input_image(image)
    clicker = StaticClicker(clickseg_click_cls, clicks)
    base_probs = predictor.get_prediction(clicker)

    result = apply_tcr(base_probs, clicks=clicks, prior_probs=prior, config=TCRConfig())
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((result.mask.astype(np.uint8) * 255)).save(args.output_mask)
    print(f"Saved {args.output_mask}")
    print(f"threshold={result.threshold:.2f}, trust={result.trust:.3f}")


if __name__ == "__main__":
    main()
