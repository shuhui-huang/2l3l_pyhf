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


def _with_fallback(func, *args):
    """Runs a fit-based function; on failure retries once with the other optimizer."""
    try:
        return func(*args)
    except Exception:
        tensorlib, optimizer = pyhf.get_backend()
        alt = (
            pyhf.optimize.scipy_optimizer()
            if isinstance(optimizer, pyhf.optimize.minuit_optimizer)
            else pyhf.optimize.minuit_optimizer(verbose=0)
        )
        pyhf.set_backend(tensorlib, alt)
        try:
            return func(*args)
        except Exception:
            return None
        finally:
            pyhf.set_backend(tensorlib, optimizer)


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
    """Test-statistic distributions for each POI value (one job).

    Same procedure as pyhf's ToyCalculator (toys thrown from the parameters
    profiled on data for the tested POI and for POI = 0), but a toy whose fit
    fails with both optimizers is dropped (and counted) instead of aborting.
    """
    from pyhf.infer.mle import fixed_poi_fit
    from pyhf.infer.utils import get_test_stat

    np.random.seed(int(seed))
    init = init_pars if init_pars is not None else model.config.suggested_init()
    bounds = par_bounds if par_bounds is not None else model.config.suggested_bounds()
    fixed = fixed_params if fixed_params is not None else model.config.suggested_fixed()
    teststat_func = get_test_stat(test_stat)
    data = pyhf.tensorlib.astensor(data)

    def ts(poi, sample):
        return _with_fallback(lambda: float(teststat_func(poi, sample, model, init, bounds, fixed)))

    bkg_pars = _with_fallback(lambda: fixed_poi_fit(1.0 if test_stat == "q0" else 0.0, data, model, init, bounds, fixed))
    if bkg_pars is None:
        raise RuntimeError("background-only fit to data failed")
    bkg_sample = model.make_pdf(bkg_pars).sample((int(ntoys),))
    out = {"poi_values": np.asarray(poi_values, dtype=float), "seed": np.asarray(seed), "test_stat": np.asarray(test_stat)}
    teststats, sb, b, failed = [], [], [], []
    for poi in poi_values:
        observed = ts(float(poi), data)
        if observed is None:
            raise RuntimeError(f"fit of the observed data failed for POI {poi}")
        teststats.append(observed)
        sig_pars = _with_fallback(lambda: fixed_poi_fit(float(poi), data, model, init, bounds, fixed))
        if sig_pars is None:
            raise RuntimeError(f"fixed-POI fit to data failed for POI {poi}")
        sig_sample = model.make_pdf(sig_pars).sample((int(ntoys),))
        sb_ts = [ts(float(poi), x) for x in sig_sample]
        b_ts = [ts(float(poi), x) for x in bkg_sample]
        n_fail = sum(t is None for t in sb_ts) + sum(t is None for t in b_ts)
        if n_fail:
            log.warning(f"POI {poi}: {n_fail} of {2 * int(ntoys)} toy fits failed and were dropped")
        failed.append(n_fail)
        sb.append(np.asarray([t for t in sb_ts if t is not None]))
        b.append(np.asarray([t for t in b_ts if t is not None]))
    out["teststat"] = np.asarray(teststats)
    # ragged if toys were dropped: store as object arrays
    out["sb"] = np.asarray(sb, dtype=object) if len({len(x) for x in sb}) > 1 else np.asarray(sb)
    out["b"] = np.asarray(b, dtype=object) if len({len(x) for x in b}) > 1 else np.asarray(b)
    out["failed"] = np.asarray(failed)
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
    loaded = [np.load(f, allow_pickle=True) for f in files]
    pois = loaded[0]["poi_values"]
    for item in loaded[1:]:
        if not np.allclose(item["poi_values"], pois):
            raise ValueError("toy files with different POI values cannot be merged")
    roles = [str(r) for r in loaded[0]["roles"]] if "roles" in loaded[0].files else None
    results: Dict[str, Any] = {"poi_values": pois.tolist(), "files": list(map(str, files)), "points": [], "roles": roles}
    obs_cls, exp_cls = [], []
    for i, poi in enumerate(pois):
        sb = np.concatenate([np.asarray(item["sb"][i], dtype=float) for item in loaded])
        b = np.concatenate([np.asarray(item["b"][i], dtype=float) for item in loaded])
        teststat = float(loaded[0]["teststat"][i])
        cls, clsb, clb, exp = _cls(teststat, sb, b)
        obs_cls.append(cls)
        exp_cls.append(exp)
        results["points"].append(
            {"poi": float(poi), "CLs": cls, "CLsb": clsb, "CLb": clb, "CLs_exp": exp, "ntoys_sb": int(len(sb)), "ntoys_b": int(len(b))}
        )
    # upper limit from the scan points (all POIs if no roles are stored); CLs = 1 at POI = 0
    scan = [i for i in range(len(pois)) if roles is None or roles[i] == "scan"]
    if len(scan) > 1 or (scan and roles is not None):
        sp = [0.0] + [float(pois[i]) for i in scan]
        so = [1.0] + [obs_cls[i] for i in scan]
        se = np.asarray([[1.0] * 5] + [exp_cls[i] for i in scan])
        results["upper_limit"] = interpolate_limit(sp, so, level)
        results["expected_upper_limit"] = [interpolate_limit(sp, se[:, k], level) for k in range(5)]
    return results


def interpolate_limit(pois: Sequence[float], cls: Sequence[float], level: float = 0.05) -> float:
    """POI value where CLs crosses ``level``.

    Interpolates log(CLs) linearly between the scan points bracketing the first
    crossing (CLs falls roughly exponentially with the POI).
    """
    pois = np.asarray(pois, dtype=float)
    cls = np.asarray(cls, dtype=float)
    good = np.isfinite(cls)
    pois, cls = pois[good], np.maximum(cls[good], 1e-12)  # CLs = 0 happens with few toys
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
