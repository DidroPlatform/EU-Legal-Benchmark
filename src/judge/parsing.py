"""Shared JSON-parsing helpers for the judge subsystem."""

from __future__ import annotations

import json
from typing import Any, Dict


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from *text*.

    Tries a direct ``json.loads`` first, then falls back to locating the
    outermost ``{…}`` pair so that models wrapping their output in markdown
    fences or extra prose still parse correctly.

    Raises ``ValueError`` when no JSON object can be found.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Response did not contain a JSON object.")
    return json.loads(text[start : end + 1])
