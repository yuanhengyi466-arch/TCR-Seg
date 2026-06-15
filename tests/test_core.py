from __future__ import annotations

import numpy as np

from tcrseg import Click, TCRConfig, apply_tcr, click_constrained_threshold


def test_click_constrained_threshold_prefers_click_consistency() -> None:
    probs = np.zeros((16, 16), dtype=np.float32) + 0.5
    probs[8, 8] = 0.62
    probs[1, 1] = 0.22
    clicks = [Click(8, 8, True), Click(1, 1, False)]
    threshold, _ = click_constrained_threshold(
        probs,
        clicks,
        default_threshold=0.49,
        mode="click_consistency_regularized",
        candidates=[0.35, 0.49, 0.60],
    )
    assert threshold in {0.49, 0.60}


def test_tcr_runs_with_prior() -> None:
    probs = np.zeros((24, 24), dtype=np.float32) + 0.45
    prior = np.zeros_like(probs) + 0.2
    prior[6:18, 6:18] = 0.85
    probs[6:18, 6:18] = 0.52
    clicks = [Click(12, 12, True), Click(2, 2, False)]
    result = apply_tcr(probs, clicks, prior, TCRConfig())
    assert result.probs.shape == probs.shape
    assert result.mask.dtype == bool
    assert result.trust > 0.9
    assert result.mask[12, 12]
