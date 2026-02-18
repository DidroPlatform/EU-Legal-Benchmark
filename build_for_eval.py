from __future__ import annotations

import importlib.util

if importlib.util.find_spec("src.data.build_for_eval") is not None:
    from src.data.build_for_eval import main
else:
    raise ModuleNotFoundError(
        "Could not resolve build module imports. Run from repo root so `src` is importable."
    )


if __name__ == "__main__":
    main()
