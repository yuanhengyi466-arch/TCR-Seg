"""TCR-Seg: Trust-Guided Context-Component Routing for click-based segmentation."""

from .clicks import Click, as_click, normalize_clicks
from .core import TCRConfig, TCRResult, apply_tcr, click_constrained_threshold, dice_iou
from .wrapper import TCRSegmenter

__all__ = [
    "Click",
    "TCRConfig",
    "TCRResult",
    "TCRSegmenter",
    "apply_tcr",
    "as_click",
    "click_constrained_threshold",
    "dice_iou",
    "normalize_clicks",
]
