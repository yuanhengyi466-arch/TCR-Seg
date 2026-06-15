"""Small integration wrapper for external interactive backbones."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .core import TCRConfig, TCRResult, apply_tcr


PredictFn = Callable[..., np.ndarray]
PriorFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class TCRSegmenter:
    """Wrap a click-conditioned 2D segmentation predictor with TCR.

    ``predict_fn`` must return a probability map in ``H x W`` format. The
    callable receives keyword arguments ``image``, ``clicks`` and ``prev_mask``.
    This keeps the core package independent from any one backbone.
    """

    predict_fn: PredictFn
    config: TCRConfig = field(default_factory=TCRConfig)
    prior_fn: PriorFn | None = None

    def predict(
        self,
        image: np.ndarray,
        clicks: Iterable[Any],
        prior_probs: np.ndarray | None = None,
        prev_mask: np.ndarray | None = None,
        **kwargs: Any,
    ) -> TCRResult:
        base_probs = self.predict_fn(image=image, clicks=clicks, prev_mask=prev_mask, **kwargs)
        if prior_probs is None and self.prior_fn is not None:
            prior_probs = self.prior_fn(image)
        return apply_tcr(base_probs, clicks=clicks, prior_probs=prior_probs, config=self.config)
