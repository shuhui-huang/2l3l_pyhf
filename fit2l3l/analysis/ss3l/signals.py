"""Signal grids of the SS/3L analysis.

A grid is described by a file in ``share/susyGridFiles/<Grid>`` listing the
datasets, one per line (HistFitter convention), e.g.::

    mc16_13TeV:mc16_13TeV.376776.MGPy8EG_A14N23LO_GG_2stepWZ_1000_550_325_100.deriv...

The masses are extracted from the physics short name with ``SigNameSchema``
(``"SP:4,LSP:5"`` = 5th and 6th ``_``-separated field), and cross-sections are
looked up in ``share/xsections/PMGxsecDB_mc16.txt``.

The signal ntuple is ``<InputPath>/signal.<lumi>.root`` with trees
``signal<DSID>_nom`` (+ ``signal<DSID>_<SYSTEMATIC>``).
"""

from __future__ import annotations

import functools
import logging
import pathlib
import re
from typing import Dict, List, Optional, Sequence, Tuple

from fit2l3l.analysis.base import SIGNAL, Sample, SignalGrid, SignalPoint
from fit2l3l.analysis.ss3l import systematics as ss3l_syst
from fit2l3l.utils import share_path

log = logging.getLogger(__name__)

#: cross-section database (PMG format)
XSEC_DB = "xsections/PMGxsecDB_mc16.txt"

#: HistFitter used xsec*kFactor (filter efficiency NOT applied); set to True to
#: multiply by the filter efficiency as well
APPLY_FILTER_EFFICIENCY = False


def parse_schema(schema: str) -> List[Tuple[str, int]]:
    """``"SP:4,LSP:5"`` -> ``[("SP", 4), ("LSP", 5)]``."""
    out = []
    for item in schema.split(","):
        label, pos = item.split(":")
        out.append((label.strip(), int(pos)))
    return out


@functools.lru_cache(maxsize=4)
def _xsec_table(path: str) -> Dict[str, List[List[str]]]:
    table: Dict[str, List[List[str]]] = {}
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "dataset_number")):
                continue
            fields = line.split()
            table.setdefault(fields[0], []).append(fields)
    return table


def lookup_xsec(dsid: str, etag: Optional[str] = None, db: Optional[str] = None):
    """Returns (xsec [fb], rel_up, rel_down, physics_short) or None if unknown."""
    path = str(share_path(db or XSEC_DB))
    if not pathlib.Path(path).exists():
        return None
    entries = _xsec_table(path).get(str(dsid))
    if not entries:
        return None
    entry = entries[0]
    if etag is not None:
        matching = [e for e in entries if e[-1] == etag]
        if matching:
            entry = matching[0]
    _, short, xsec, feff, kfac, rel_up, rel_down = entry[:7]
    value = float(xsec) * float(kfac) * 1000.0  # pb -> fb
    if APPLY_FILTER_EFFICIENCY:
        value *= float(feff)
    rel_up, rel_down = float(rel_up), float(rel_down)
    if rel_down == 0.0:
        rel_down = rel_up  # symmetrise if no down uncertainty is given
    return value, rel_up, rel_down, short


def read_grid(grid: str, sig_name_schema: str = "SP:4,LSP:5", exclude: Sequence[str] = ()) -> SignalGrid:
    """Reads ``share/susyGridFiles/<grid>``."""
    path = share_path(f"susyGridFiles/{grid}")
    if not path.exists():
        raise FileNotFoundError(f"signal grid file {path} not found")
    schema = parse_schema(sig_name_schema)
    points: List[SignalPoint] = []
    seen_masses = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("mc") or "#" in line:
            continue
        fields = line.split(".")
        dsid, short = fields[1], fields[2]
        if dsid in exclude or line in exclude:
            continue
        parts = short.split("_")
        masses = tuple(float(parts[pos].replace("p", ".")) for _, pos in schema)
        if masses in seen_masses:
            log.warning(
                f"grid {grid}: DSID {dsid} has the same masses {masses} as {seen_masses[masses]}"
            )
        seen_masses[masses] = dsid
        etag = next((t for f in fields for t in f.split("_") if re.fullmatch(r"e\d+", t)), None)
        point = SignalPoint(dsid=dsid, masses=masses, physics_short=short)
        xs = lookup_xsec(dsid, etag)
        if xs is not None:
            point.xsec, point.xsec_rel_up, point.xsec_rel_down, _ = xs
        else:
            log.debug(f"no cross-section found for DSID {dsid}")
        point.extra_weights = extra_signal_weights(grid, point)
        points.append(point)
    labels = tuple(label for label, _ in schema)
    return SignalGrid(name=grid, points=points, mass_labels=labels)


def extra_signal_weights(grid: str, point: SignalPoint) -> List[str]:
    """Grid-specific extra weights (HistFitter: GGM higgsino BR re-weighting)."""
    if grid == "Mc16SusyC1N2N1_GGMHinoZh" and int(point.m12) != 50:
        return [f"GGM_BR{int(point.m12)}Weight"]
    return []


def signal_sample(point: SignalPoint, grid: SignalGrid, options: Optional[dict] = None) -> Sample:
    """Sample definition of a signal point."""
    options = options or {}
    sample = Sample(
        name=point.name,
        kind=SIGNAL,
        color="#e7298a",
        label=f"{grid.name} ({', '.join(f'{m:g}' for m in point.masses)})",
        file="signal.{lumi}.root",
        tree=f"{point.name}_nom",
        tree_prefix=point.name,
        extra_weights=list(point.extra_weights),
        extra_systematics=list(ss3l_syst.SIGNAL_THEORY),
        meta={"dsid": point.dsid, "masses": list(point.masses)},
    )
    if options.get("isRPCmRPV"):
        sample.exclude_systematics = list(ss3l_syst.RPCMRPV_SIGNAL_EXCLUDE)
        sample.extra_systematics += list(ss3l_syst.RPCMRPV_SIGNAL_EXTRA)
    return sample
