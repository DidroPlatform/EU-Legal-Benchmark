from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent

# Ensure imports work whether tests run from this repo root or its parent.
for path in (str(ROOT), str(PARENT)):
    if path not in sys.path:
        sys.path.insert(0, path)
