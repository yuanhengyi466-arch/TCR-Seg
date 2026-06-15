# TCR-Seg

This repository contains a lightweight inference controller for click-based 2D medical image segmentation. It wraps an external interactive segmentation backbone, such as a FocalClick/ClickSEG-style predictor, and updates each click response with:

- prior-click trust modulation;
- trust-guided logit update around uncertain pixels;
- click-constrained threshold selection;
- component-aware routing utilities for benchmark click simulation.

The repository does not include datasets, checkpoints, experiment logs, or vendored baseline code.

## Installation

```bash
pip install -e ".[demo,test]"
```

For the minimal runtime only:

```bash
pip install -e .
```

## Quick Smoke Test

```bash
python examples/demo_numpy.py
pytest -q
```

The demo creates a synthetic image, cached prior, click list, and toy click-conditioned predictor. It writes a visual check to `examples/outputs/demo_numpy.png`.

## Core Usage

```python
import numpy as np
from tcrseg import Click, TCRConfig, apply_tcr

pred_probs = np.random.rand(256, 256).astype("float32")
prior_probs = np.random.rand(256, 256).astype("float32")
clicks = [Click(y=120, x=134, positive=True), Click(y=30, x=40, positive=False)]

result = apply_tcr(
    pred_probs,
    clicks=clicks,
    prior_probs=prior_probs,
    config=TCRConfig(),
)

mask = result.mask
print(result.threshold, result.trust)
```

## Backbone Integration

TCR-Seg expects the backbone to output a 2D probability map after each click. The wrapper API is deliberately small:

```python
from tcrseg import TCRSegmenter

def predict_fn(image, clicks, prev_mask=None):
    # Call your FocalClick / ClickSEG / SimpleClick / custom model here.
    return probability_map

segmenter = TCRSegmenter(predict_fn=predict_fn)
result = segmenter.predict(image, clicks, prior_probs=cached_prior)
```

An optional ClickSEG/FocalClick adapter template is provided in `examples/run_clickseg_tcr.py`. It requires an external ClickSEG repository and checkpoint path.

## Default Settings

- `prior_mode`: `uncertain_trust_logit`
- `prior_weight`: `0.10`
- `prior_start_click`: `1`
- `prior_click_margin`: `0.05`
- `prior_trust_tau`: `0.15`
- `threshold_mode`: `click_consistency_regularized`
- `threshold_candidates`: `[0.35, 0.40, 0.45, 0.49, 0.50, 0.55, 0.60, 0.65]`
- benchmark routing strategy: `distance_then_component_area` with switch after 9 clicks

The same values are also listed in `configs/tcrseg_default.yaml`.

## Repository Layout

```text
tcrseg/
  clicks.py              # Framework-agnostic click objects
  core.py                # Prior-click trust, logit update, thresholding
  component_routing.py   # Component-aware click simulation and mask utilities
  wrapper.py             # Small wrapper for external predictors
examples/
  demo_numpy.py          # No-checkpoint synthetic smoke demo
  run_clickseg_tcr.py    # Optional ClickSEG/FocalClick adapter template
configs/
  tcrseg_default.yaml
tests/
  test_core.py
```

## Benchmarking Note

Component-aware next-click routing uses ground-truth masks and is intended only for standard offline benchmark click simulation. Real interactive use should receive clicks from a user interface.
