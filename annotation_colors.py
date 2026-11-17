"""Shared RGB colors for review, drawing, and rendered video annotations."""

RED = (255, 60, 60)
GREEN = (60, 200, 90)
UNKNOWN_SENSITIVE = (255, 128, 0)
UNKNOWN_NON_SENSITIVE = (46, 196, 166)
HIGHLIGHT = (74, 163, 255)


def color_for(sensitivity: str, class_name: str | None = None) -> tuple[int, int, int]:
    """Keep semantic uncertainty visible without hiding the sensitivity decision."""
    safe = sensitivity == "non_sensitive"
    if (class_name or "").strip().lower() == "unknown":
        return UNKNOWN_NON_SENSITIVE if safe else UNKNOWN_SENSITIVE
    return GREEN if safe else RED


def color_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)
