"""Small helpers shared by the workspace builder and the fit runner."""

from __future__ import annotations

import importlib
import json
import logging
import os
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import yaml

log = logging.getLogger("fit2l3l")

#: repository root (one level above the package)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
#: directory with shared inputs (grids, cross-sections, hard-coded yields ...)
SHARE_DIR = pathlib.Path(os.environ.get("FIT2L3L_SHARE", REPO_ROOT / "share"))


def share_path(relative: Union[str, pathlib.Path]) -> pathlib.Path:
    path = pathlib.Path(relative)
    if path.is_absolute() or path.exists():
        return path
    return SHARE_DIR / path


def setup_logging(verbosity: Union[int, str] = "INFO") -> None:
    if isinstance(verbosity, int):
        level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    else:
        level = getattr(logging, str(verbosity).upper())
    logging.basicConfig(level=level, format="%(levelname)-7s %(name)s: %(message)s")
    for noisy in ("matplotlib", "PIL", "numba", "jax"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("cabinetry").setLevel(max(level, logging.WARNING) if level > logging.DEBUG else level)
    # "histogram has empty bins" warnings are expected for small backgrounds
    logging.getLogger("cabinetry.histo").setLevel(logging.ERROR if level > logging.DEBUG else level)
    logging.getLogger("pyhf").setLevel(logging.WARNING)


def load_yaml(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(obj: Any, path: Union[str, pathlib.Path], header: str = "") -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        if header:
            for line in header.strip().splitlines():
                handle.write(f"# {line}\n")
        yaml.safe_dump(to_builtin(obj), handle, sort_keys=False, width=120)


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pathlib.Path):
            return str(o)
        return super().default(o)


def to_builtin(obj: Any) -> Any:
    return json.loads(json.dumps(obj, cls=_Encoder))


def write_json(obj: Any, path: Union[str, pathlib.Path], indent: Optional[int] = 2) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, cls=_Encoder, indent=indent, sort_keys=False))


def read_json(path: Union[str, pathlib.Path]) -> Any:
    return json.loads(pathlib.Path(path).read_text())


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [value]


def load_analysis(name: str, options: Optional[Dict[str, Any]] = None):
    """Imports an analysis module and returns its :class:`Analysis` object.

    ``name`` is either a sub-module of :mod:`fit2l3l.analysis` (e.g. ``ss3l``) or a
    full dotted module path.  The module must provide ``build_analysis(options)``.
    """
    module_name = name if "." in name else f"fit2l3l.analysis.{name}"
    module = importlib.import_module(module_name)
    return module.build_analysis(options or {})


def unique(items: Iterable[Any]) -> List[Any]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
