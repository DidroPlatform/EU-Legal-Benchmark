from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
RUNS_ROOT = ROOT / "data" / "runs"
_BASELINE_RUN_DIRS: set[Path] = set()


def _direct_child_dirs(path: Path) -> set[Path]:
    if not path.exists():
        return set()
    return {child.resolve() for child in path.iterdir() if child.is_dir()}


# Ensure imports work whether tests run from this repo root or its parent.
for path in (str(ROOT), str(PARENT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_sessionstart(session: object) -> None:
    del session
    global _BASELINE_RUN_DIRS
    _BASELINE_RUN_DIRS = _direct_child_dirs(RUNS_ROOT)


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    del session, exitstatus
    created_this_session = _direct_child_dirs(RUNS_ROOT) - _BASELINE_RUN_DIRS
    for run_dir in sorted(created_this_session):
        try:
            shutil.rmtree(run_dir)
        except FileNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover - defensive cleanup warning
            warnings.warn(
                f"Failed to remove pytest-created run directory '{run_dir}': {exc}",
                RuntimeWarning,
                stacklevel=1,
            )
