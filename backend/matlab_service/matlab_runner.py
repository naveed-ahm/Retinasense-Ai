"""Safe, optional bridge from RetinaSense AI to the local MATLAB Engine."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger("retinasense.matlab")

_engine: Any | None = None
_engine_attempted = False
_engine_lock = Lock()
_MATLAB_DIR = Path(__file__).resolve().parents[1] / "matlab"


class MatlabUnavailable(RuntimeError):
    """Raised only inside this optional add-on when MATLAB Engine is unavailable."""


def _get_engine() -> Any:
    global _engine, _engine_attempted
    with _engine_lock:
        if _engine is not None:
            return _engine
        if _engine_attempted:
            raise MatlabUnavailable("MATLAB Engine is unavailable")
        _engine_attempted = True
        try:
            import matlab.engine  # type: ignore[import-not-found]
            engine = matlab.engine.start_matlab()
            engine.addpath(str(_MATLAB_DIR), nargout=0)
            _engine = engine
            return engine
        except Exception as exc:
            logger.info("MATLAB add-on unavailable: %s", exc)
            raise MatlabUnavailable("MATLAB Engine is unavailable") from exc


def is_matlab_available() -> bool:
    try:
        _get_engine()
        return True
    except MatlabUnavailable:
        return False


def _json_result(function: str, *args: str) -> dict[str, Any]:
    engine = _get_engine()
    try:
        raw = engine.feval(function, *args, nargout=1)
        return json.loads(str(raw))
    except MatlabUnavailable:
        raise
    except Exception as exc:
        logger.exception("MATLAB %s failed", function)
        raise RuntimeError("MATLAB image analysis failed") from exc


def run_quality_analysis(image_path: str | Path) -> dict[str, Any]:
    return _json_result("fundus_quality", str(Path(image_path).resolve()))


def run_enhancement(image_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    return _json_result("enhance_fundus", str(Path(image_path).resolve()), str(Path(output_path).resolve()))


def run_retinal_features(image_path: str | Path) -> dict[str, Any]:
    return _json_result("retinal_features", str(Path(image_path).resolve()))
