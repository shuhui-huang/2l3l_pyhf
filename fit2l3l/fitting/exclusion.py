"""Exclusion fits over a signal grid (HistFitter ``-F excl -p [-l]``).

For every signal point (a patch of ``patchset.json`` applied to ``ws.json``):

* CLs (observed, expected and the +-1,2 sigma band) for mu_SIG = 1
* observed CLs for the signal cross-section varied by +-1 sigma (theory band of the
  observed contour; done by testing mu = 1 +- dsigma/sigma)
* optional upper limit on mu_SIG (observed + expected band) and the excluded
  cross-section
* optional toys (``Calculator: toybased``), split over jobs with ``Seed`` and
  merged with ``fit2l3l merge``

Results are written per point to ``<out>/points/<point>.json`` (HistFitter
hypotest JSON names: CLs, CLsexp, clsd1s, ..., upperLimit, m0, m12, fID, xsec) and
merged into ``<out>/hypotest_list.json`` for the contour plots.
"""

from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing
import pathlib
import traceback
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pyhf

from fit2l3l.fitting import common, discovery, toys
from fit2l3l.utils import as_list, read_json, write_json

log = logging.getLogger(__name__)

EXPECTED_KEYS = ["clsd2s", "clsd1s", "CLsexp", "clsu1s", "clsu2s"]
UL_KEYS = [
    "expectedUpperLimitMinus2Sig",
    "expectedUpperLimitMinus1Sig",
    "expectedUpperLimit",
    "expectedUpperLimitPlus1Sig",
    "expectedUpperLimitPlus2Sig",
]


def _point_info(ctx: common.WorkspaceContext, patch) -> Dict[str, Any]:
    info = {"name": patch.name, "values": list(patch.values)}
    meta_points = {p["name"]: p for p in ctx.meta.get("signal", {}).get("points", [])}
    md = patch.metadata if hasattr(patch, "metadata") else {}
    mp = meta_points.get(patch.name, {})
    info["dsid"] = md.get("dsid", mp.get("dsid", patch.name.replace("signal", "")))
    info["xsec"] = md.get("xsec_fb", mp.get("xsec"))
    info["xsec_rel_up"] = md.get("xsec_rel_up", mp.get("xsec_rel_up"))
    info["xsec_rel_down"] = md.get("xsec_rel_down", mp.get("xsec_rel_down"))
    return info


def _hypotest(model, data, mu, fit_cfg, kwargs, expected=True):
    return pyhf.infer.hypotest(
        mu, data, model, test_stat=fit_cfg.get("TestStat", "qtilde"), return_expected_set=expected, **kwargs
    )


def run_point(
    ctx: common.WorkspaceContext,
    general: Dict[str, Any],
    fit_cfg: Dict[str, Any],
    patch_name: str,
    outdir: pathlib.Path,
    fit_regions: Sequence[str],
    blinded: Sequence[str],
) -> Dict[str, Any]:
    common.setup_backend(general)
    patch = ctx.patchset[patch_name]
    info = _point_info(ctx, patch)
    ws = ctx.patchset.apply(ctx.workspace, patch_name)
    signal_yields = {
        c["name"]: float(np.sum(s["data"])) for c in ws["channels"] for s in c["samples"] if s["name"] == patch_name
    }
    _, model, data = common.build_model(dict(ws), general, channels=fit_regions)
    if fit_cfg.get("Asimov"):
        init, _, _ = common.parameter_settings(model, fit_cfg)
        data = list(common.asimov_background(model, init)) + list(model.config.auxdata)
    else:
        data = common.blind_data(model, data, [b for b in blinded if b in fit_regions])
    kwargs = discovery.hypotest_kwargs(model, fit_cfg)
    seed = int(fit_cfg.get("Seed", 0))
    result: Dict[str, Any] = {
        "name": patch_name,
        "fID": info["dsid"],
        "masses": info["values"],
        "m0": float(info["values"][0]) if info["values"] else float("nan"),
        "m12": float(info["values"][1]) if len(info["values"]) > 1 else 0.0,
        "xsec": info["xsec"],
        "xsec_rel_up": info["xsec_rel_up"],
        "xsec_rel_down": info["xsec_rel_down"],
        "calctype": kwargs["calctype"],
        "regions": list(fit_regions),
        "signal_yields": signal_yields,  # pre-fit signal yields per region (CollectSignalYields)
        "failedfit": 0,
    }
    if kwargs["calctype"] == "toybased":
        result.update({"ntoys": kwargs["ntoys"], "seed": seed})
    saved_toys = kwargs["calctype"] == "toybased" and fit_cfg.get("SaveToys")
    try:
        if saved_toys:
            # one set of toys per POI: mu = 1, the theory variations and the upper-limit
            # scan; jobs with different seeds are merged by `fit2l3l merge`
            pois, roles = [1.0], ["nominal"]
            if fit_cfg.get("TheoryVariations", True) and info.get("xsec_rel_up") is not None:
                pois += [1.0 + float(info["xsec_rel_up"]), max(1.0 - float(info["xsec_rel_down"]), 1e-3)]
                roles += ["theory_up", "theory_down"]
            ul_cfg = fit_cfg.get("UpperLimit") or {}
            if ul_cfg.get("Enabled", True):
                scan = ul_cfg.get("Scan")
                if isinstance(scan, (list, tuple)) and len(scan) == 3:
                    grid = [float(x) for x in np.linspace(float(scan[0]), float(scan[1]), int(scan[2])) if x > 0]
                    pois += grid
                    roles += ["scan"] * len(grid)
                else:
                    log.warning(f"{patch_name}: toy upper limits need UpperLimit.Scan: [low, high, n]; skipped")
            dist = toys.generate(model, data, pois, ntoys=kwargs["ntoys"], seed=seed,
                                 test_stat=fit_cfg.get("TestStat", "qtilde"), init_pars=kwargs["init_pars"],
                                 fixed_params=kwargs["fixed_params"], par_bounds=kwargs["par_bounds"])
            dist["roles"] = np.asarray(roles)
            toy_file = outdir / "toys" / f"toys_{patch_name}_seed{seed}.npz"
            toys.save(dist, toy_file)
            apply_toy_results(result, toys.merge([str(toy_file)], 1.0 - float(fit_cfg.get("CL", 0.95))))
        else:
            if kwargs["calctype"] == "toybased":
                np.random.seed(seed)
            cls_obs, cls_exp = _hypotest(model, data, 1.0, fit_cfg, kwargs)
            result["CLs"] = float(cls_obs)
            for key, value in zip(EXPECTED_KEYS, cls_exp):
                result[key] = float(value)
    except Exception as exc:
        log.error(f"{patch_name}: hypotest failed: {exc}")
        log.debug(traceback.format_exc())
        result["failedfit"] = 1
        result["CLs"] = float("nan")
        for key in EXPECTED_KEYS:
            result[key] = float("nan")

    # observed CLs with the signal cross-section varied by its uncertainty
    if (fit_cfg.get("TheoryVariations", True) and info.get("xsec_rel_up") is not None and not result["failedfit"]
            and not saved_toys):
        for key, mu in (("CLs_theory_up", 1.0 + float(info["xsec_rel_up"])), ("CLs_theory_down", 1.0 - float(info["xsec_rel_down"]))):
            try:
                if kwargs["calctype"] == "toybased":
                    np.random.seed(seed)
                result[key] = float(_hypotest(model, data, max(mu, 1e-3), fit_cfg, kwargs, expected=False))
            except Exception as exc:
                log.warning(f"{patch_name}: theory-varied hypotest failed: {exc}")
                result[key] = float("nan")

    # upper limit on mu_SIG
    if (fit_cfg.get("UpperLimit") or {}).get("Enabled", True) and not result["failedfit"] and not saved_toys:
        try:
            if kwargs["calctype"] == "toybased":
                np.random.seed(seed)
            obs, exp, method = discovery.upper_limit(model, data, fit_cfg, kwargs)
            result["upperLimit"] = obs
            for key, value in zip(UL_KEYS, exp):
                result[key] = value
            result["upperLimitMethod"] = method
            if info["xsec"]:
                result["excludedXsec"] = obs * info["xsec"]
                result["expectedExcludedXsec"] = exp[2] * info["xsec"]
        except Exception as exc:
            log.warning(f"{patch_name}: upper limit failed: {exc}")
            result["upperLimit"] = float("nan")

    if fit_cfg.get("PointFits"):
        import cabinetry

        try:
            init, fixed, bounds = common.parameter_settings(model, fit_cfg)
            res = cabinetry.fit.fit(model, data, init_pars=init, fix_pars=fixed, par_bounds=bounds)
            common.save_fit_results(res, outdir / "point_fits" / f"{patch_name}.json")
            result["mu_hat"] = float(res.bestfit[model.config.poi_index])
        except Exception as exc:
            log.warning(f"{patch_name}: free fit failed: {exc}")

    write_json(result, outdir / "points" / f"{patch_name}.json")
    log.info(
        f"{patch_name} {tuple(info['values'])}: CLs = {result['CLs']:.4g}, CLsexp = {result['CLsexp']:.4g}"
        + (f", UL = {result['upperLimit']:.3g}" if "upperLimit" in result else "")
    )
    return result


def apply_toy_results(result: Dict[str, Any], merged: Dict[str, Any]) -> None:
    """Fills CLs, theory-varied CLs and upper limits of a point from merged toys."""
    roles = merged.get("roles") or ["nominal"] + ["scan"] * (len(merged["points"]) - 1)
    for role, p in zip(roles, merged["points"]):
        if role == "nominal":
            result["CLs"] = p["CLs"]
            for key, value in zip(EXPECTED_KEYS, p["CLs_exp"]):
                result[key] = value
            result["ntoys"] = p["ntoys_b"]
        elif role == "theory_up":
            result["CLs_theory_up"] = p["CLs"]
        elif role == "theory_down":
            result["CLs_theory_down"] = p["CLs"]
    result["toy_files"] = len(merged["files"])
    if "upper_limit" in merged:
        result["upperLimit"] = merged["upper_limit"]
        for key, value in zip(UL_KEYS, merged["expected_upper_limit"]):
            result[key] = value
        result["upperLimitMethod"] = "toys"
        if result.get("xsec"):
            result["excludedXsec"] = result["upperLimit"] * result["xsec"]
            result["expectedExcludedXsec"] = result["expectedUpperLimit"] * result["xsec"]


def select_points(ctx: common.WorkspaceContext, points: Any) -> List[str]:
    if ctx.patchset is None:
        raise RuntimeError(f"{ctx.directory} has no patchset.json - was the workspace built with FitType: excl?")
    names = [p.name for p in ctx.patchset.patches]
    if points in (None, "all", "ALL"):
        return names
    wanted = []
    for p in as_list(points):
        p = str(p)
        match = [n for n in names if n == p or n == f"signal{p}"]
        if not match:
            raise KeyError(f"signal point {p} not in the patchset")
        wanted += match
    return wanted


def _worker(args):
    ctx, general, fit_cfg, name, outdir, fit_regions, blinded = args
    try:
        return run_point(ctx, general, fit_cfg, name, outdir, fit_regions, blinded)
    except Exception as exc:  # pragma: no cover
        log.error(f"{name}: {exc}")
        return {"name": name, "failedfit": 1, "error": str(exc)}


def run(
    ctx: common.WorkspaceContext,
    general: Dict[str, Any],
    fit_cfg: Dict[str, Any],
    outdir: pathlib.Path,
    *,
    contours: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
    do_merge: bool = True,
) -> Dict[str, Any]:
    names = select_points(ctx, fit_cfg.get("Points"))
    fit_regions, val_regions = common.fit_and_validation_regions(ctx, fit_cfg)
    blinded = common.resolve_blinded(ctx, fit_cfg.get("Blind"))
    if val_regions:
        log.info(f"regions not used in the exclusion fit: {val_regions}")
    log.info(f"exclusion fits for {len(names)} signal points in {fit_regions}")
    n_jobs = int(fit_cfg.get("Jobs", n_jobs) or 1)
    args = [(ctx, general, fit_cfg, n, outdir, fit_regions, blinded) for n in names]
    if n_jobs > 1 and len(names) > 1:
        with concurrent.futures.ProcessPoolExecutor(n_jobs, mp_context=multiprocessing.get_context("fork")) as pool:
            results = list(pool.map(_worker, args))
    else:
        results = [_worker(a) for a in args]
    merged = merge(outdir, ctx) if do_merge else []
    if contours and merged:
        from fit2l3l.plotting import contours as contour_plots

        contour_plots.plot_from_config(contours, merged_file=outdir / "hypotest_list.json", meta=ctx.meta,
                                       outdir=outdir / "plots", general=general)
    return {"points": len(results), "failed": [r["name"] for r in results if r.get("failedfit")],
            "merged": str(outdir / "hypotest_list.json") if merged else None}


def merge(outdir: pathlib.Path, ctx: Optional[common.WorkspaceContext] = None, *, toys_dir: bool = True) -> List[Dict[str, Any]]:
    """Merges ``points/*.json`` (and toy files, if any) into ``hypotest_list.json``."""
    outdir = pathlib.Path(outdir)
    if toys_dir and (outdir / "toys").exists():
        merged_toys = toys.merge_directory(outdir / "toys")
        for tag, res in merged_toys.items():
            name = tag[len("toys_"):] if tag.startswith("toys_") else tag
            point_file = outdir / "points" / f"{name}.json"
            if not point_file.exists():
                continue
            point = read_json(point_file)
            apply_toy_results(point, res)
            write_json(point, point_file)
    points = []
    for f in sorted((outdir / "points").glob("*.json")):
        points.append(read_json(f))
    if not points:
        log.warning(f"no point results found in {outdir / 'points'}")
        return []
    write_json(points, outdir / "hypotest_list.json")
    missing = []
    if ctx is not None and ctx.patchset is not None:
        done = {p["name"] for p in points}
        missing = [p.name for p in ctx.patchset.patches if p.name not in done]
        (outdir / "missing_points.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    failed = [p["name"] for p in points if p.get("failedfit")]
    log.info(f"merged {len(points)} points into {outdir / 'hypotest_list.json'}"
             + (f", {len(missing)} missing (missing_points.txt)" if missing else "")
             + (f", {len(failed)} failed" if failed else ""))
    return points
