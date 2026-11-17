"""Local Streamlit component for live annotation drawing; no external JS service."""

import base64
import io
from pathlib import Path

import streamlit.components.v1 as components
from PIL import Image

from annotation_colors import UNKNOWN_NON_SENSITIVE, UNKNOWN_SENSITIVE, color_hex

_canvas = components.declare_component(
    "annotation_drawing_canvas", path=str(Path(__file__).parent / "drawing_canvas")
)


def drawing_canvas(
    image: Image.Image, *, key: str, background_color: str,
    text_color: str, surface_color: str, initial_sensitivity: str,
) -> dict | None:
    """Return one saved normalized outline; pointer moves stay in the browser."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return _canvas(
        image_url="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        image_width=image.width, image_height=image.height,
        sensitive_color=color_hex(UNKNOWN_SENSITIVE),
        non_sensitive_color=color_hex(UNKNOWN_NON_SENSITIVE),
        background_color=background_color, text_color=text_color, surface_color=surface_color,
        initial_sensitivity=initial_sensitivity,
        key=key, default=None,
    )
