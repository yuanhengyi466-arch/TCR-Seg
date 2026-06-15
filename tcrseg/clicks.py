"""Click primitives used by TCR-Seg.

The core package intentionally keeps clicks independent from any specific
interactive segmentation framework. Adapters can convert these objects to the
format expected by FocalClick, ClickSEG, SimpleClick, or other backbones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Click:
    """A positive or negative user click in ``(y, x)`` image coordinates."""

    y: int
    x: int
    positive: bool = True
    index: int | None = None

    @property
    def coords(self) -> tuple[int, int]:
        return int(self.y), int(self.x)

    @property
    def is_positive(self) -> bool:
        return bool(self.positive)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "y": int(self.y),
            "x": int(self.x),
            "positive": bool(self.positive),
        }
        if self.index is not None:
            out["index"] = int(self.index)
        return out


def as_click(value: Any) -> Click:
    """Convert common click representations into :class:`Click`.

    Accepted inputs include ``Click``, dictionaries with ``y``/``x`` fields,
    dictionaries with ``coords``, and framework objects exposing ``coords`` and
    ``is_positive`` attributes.
    """

    if isinstance(value, Click):
        return value

    if isinstance(value, dict):
        positive = value.get("positive", value.get("is_positive", True))
        index = value.get("index", value.get("indx", None))
        if "coords" in value:
            y, x = value["coords"]
        else:
            y, x = value["y"], value["x"]
        return Click(int(round(y)), int(round(x)), bool(positive), None if index is None else int(index))

    if hasattr(value, "coords"):
        y, x = value.coords
        positive = getattr(value, "is_positive", getattr(value, "positive", True))
        index = getattr(value, "indx", getattr(value, "index", None))
        return Click(int(round(y)), int(round(x)), bool(positive), None if index is None else int(index))

    raise TypeError(f"Unsupported click type: {type(value)!r}")


def normalize_clicks(clicks: Iterable[Any] | None) -> list[Click]:
    if clicks is None:
        return []
    return [as_click(click) for click in clicks]
