"""Toy-based hypothesis tests that can be split into many jobs and merged.

HistFitter users split large numbers of toys over batch jobs with different
random seeds (``toys_helper``).  The same is done here: every job evaluates the
test-statistic distributions for a list of POI values with its own seed and
saves them (``toys_<tag>_seed<N>.npz``); :func:`merge` combines all files and
computes CLs (observed and expected band) and, for several POI values, the
upper limit.
"""

from __future__ import annotations

import glob
import logging
import pathlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pyhf

log = logging.getLogger(__name__)


def generate(
    model,
    data,
    poi_values: Sequence[float],
    *,
    ntoys: int,
    seed: int,
    test_stat: str = "qtilde",
    init_pars=None,
    fixed_params=None,
    par_bounds=None,
) -> Dict[str, np.ndarray]:
    """Test statistic distributions for each POI value (one job)."""
    np.random.seed(int(seed))
    calc = pyhf.infer.calculators.ToyCalculator(
        data,
        model,
        init_pars=init_pars,
        par_bounds=par_bounds,
        fixed_params=fixed_params,
        test_stat=test_stat,
        ntoys=int(ntoys),
        track_progress=False,
    )
    out = {"poi_values": np.asarray(poi_values, dtype=float), "seed": np.asarray(seed), "test_stat": np.asarray(test_stat)}
    teststats, sb, b = [], [], []
    for poi in poi_values:
        teststats.append(float(calc.teststatistic(float(poi))))
        sb_dist, b_dist = calc.distributions(float(poi))
        sb.append(np.asarray(sb_dist.samples))
        b.append(np.asarray(b_dist.samples))
    out["teststat"] = np.asarray(teststats)
    out["sb"] = np.asarray(sb)
    out["b"] = np.asarray(b)
    return out


def save(result: Dict[str, np.ndarray], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **result)


def _pvalue(sorted_samples: np.ndarray, values) -> np.ndarray:
    """Fraction of samples >= value (pyhf EmpiricalDistribution.pvalue), vectorised."""
    n = len(sorted_samples)
    return (n - np.searchsorted(sorted_samples, np.asarray(values, dtype=float), side="left")) / n


def _cls(teststat: float, sb: np.ndarray, b: np.ndarray):
    """Observed CLs and expected band, same conventions as pyhf's ToyCalculator."""
    sb_sorted = np.sort(np.asarray(sb, dtype=float))
    b_sorted = np.sort(np.asarray(b, dtype=float))
    clsb = float(_pvalue(sb_sorted, teststat))
    clb = float(_pvalue(b_sorted, teststat))
    cls = clsb / clb if clb > 0 else float("nan")
    # expected band: percentiles of CLs evaluated on the background-only toys
    with np.errstate(divide="ignore", invalid="ignore"):
        cls_b = _pvalue(sb_sorted, b_sorted) / _pvalue(b_sorted, b_sorted)
    exp = np.percentile(cls_b[np.isfinite(cls_b)], [2.27, 15.87, 50.0, 84.13, 97.72]).tolist()
    return cls, clsb, clb, exp


def merge(files: Sequence[str], level: float = 0.05) -> Dict[str, Any]:
    """Merges toy files of the same POI scan; returns CLs per POI (+ upper limit)."""
    files = sorted(files)
    if not files:
        raise FileNotFoundError("no toy files to merge")
    loaded = [np.load(f) for f in files]
    pois = loaded[0]["poi_values"]
    for item in loaded[1:]:
        if not np.allclose(item["poi_values"], pois):
            raise ValueError("toy files with different POI values cannot be merged")
    results: Dict[str, Any] = {"poi_values": pois.tolist(), "files": list(map(str, files)), "points": []}
    obs_cls, exp_cls = [], []
    for i, poi in enumerate(pois):
        sb = np.concatenate([item["sb"][i] for item in loaded])
        b = np.concatenate([item["b"][i] for item in loaded])
        teststat = float(loaded[0]["teststat"][i])
        cls, clsb, clb, exp = _cls(teststat, sb, b)
        obs_cls.append(cls)
        exp_cls.append(exp)
        results["points"].append(
            {"poi": float(poi), "CLs": cls, "CLsb": clsb, "CLb": clb, "CLs_exp": exp, "ntoys_sb": int(len(sb)), "ntoys_b": int(len(b))}
        )
    if len(pois) > 1:
        results["upper_limit"] = interpolate_limit(pois, obs_cls, level)
        exp_arr = np.asarray(exp_cls)
        results["expected_upper_limit"] = [interpolate_limit(pois, exp_arr[:, k], level) for k in range(5)]
    return results


def interpolate_limit(pois: Sequence[float], cls: Sequence[float], level: float = 0.05) -> float:
    """POI value where CLs crosses ``level``.

    Interpolates log(CLs) linearly between the scan points bracketing the first
    crossing (CLs falls roughly exponentially with the POI).
    """
    pois = np.asarray(pois, dtype=float)
    cls = np.asarray(cls, dtype=float)
    good = np.isfinite(cls) & (cls > 0)
    pois, cls = pois[good], cls[good]
    below = np.where(cls < level)[0]
    if len(below) == 0:
        return float("nan")
    j = below[0]
    if j == 0:
        return float(pois[0])
    x0, x1 = pois[j - 1], pois[j]
    y0, y1 = np.log(cls[j - 1]), np.log(cls[j])
    if y1 == y0:
        return float(x1)
    return float(x0 + (np.log(level) - y0) * (x1 - x0) / (y1 - y0))


def merge_directory(directory: pathlib.Path, pattern: str = "toys_*.npz", level: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Groups ``toys_<tag>_seed<N>.npz`` files by tag and merges each group."""
    groups: Dict[str, List[str]] = {}
    for f in glob.glob(str(directory / pattern)):
        tag = pathlib.Path(f).stem.rsplit("_seed", 1)[0]
        groups.setdefault(tag, []).append(f)
    return {tag: merge(files, level) for tag, files in sorted(groups.items())}
