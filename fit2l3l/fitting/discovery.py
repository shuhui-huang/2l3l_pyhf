"""Discovery / model-independent fit (HistFitter ``-F disc`` + ``UpperLimitTable.py``).

The workspace contains a dummy signal of one event in the (single-bin) SR scaled
by the POI, so the POI is the number of signal events.  Computed:

* discovery p-value p(s=0) and significance Z (test statistic q0)
* upper limit on the number of signal events S95 (observed, expected +-1,2 sigma)
* visible cross-section <eps sigma>95 = S95_obs / L
* CLb at the observed limit
* the free fit and its post-fit outputs (same as background-only)
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict

import numpy as np
import pyhf
from scipy.stats import norm

from fit2l3l.fitting import bkgonly, common, toys
from fit2l3l.utils import read_json, write_json

log = logging.getLogger(__name__)


def hypotest_kwargs(model, fit_cfg) -> Dict[str, Any]:
    init, fixed, bounds = common.parameter_settings(model, fit_cfg)
    kwargs = dict(init_pars=init, fixed_params=fixed, par_bounds=bounds, calctype=fit_cfg.get("Calculator", "asymptotics"))
    if kwargs["calctype"] == "toybased":
        kwargs["ntoys"] = int(fit_cfg.get("NToys", 1000))
        kwargs["track_progress"] = False
    return kwargs


def upper_limit(model, data, fit_cfg, kwargs, *, max_poi: float = 1e4):
    """Observed and expected (-2..+2 sigma) upper limits on the POI.

    ``UpperLimit.Scan``:

    * ``null`` (default): HistFitter-like linear scan of ``NPoints`` values in
      [0, mu_max]; mu_max is found by doubling until observed and +2 sigma
      expected CLs are below the level (CLs interpolated linearly)
    * ``[low, high, n]``: fixed linear scan
    * ``toms748``: root finding of pyhf (precise, more hypothesis tests)
    """
    level = 1.0 - float(fit_cfg.get("CL", 0.95))
    ul_cfg = fit_cfg.get("UpperLimit") or {}
    scan = ul_cfg.get("Scan")
    test_stat = fit_cfg.get("TestStat", "qtilde")
    ipoi = model.config.poi_index
    kwargs = dict(kwargs)
    bounds = [list(b) for b in (kwargs.get("par_bounds") or model.config.suggested_bounds())]
    bounds[ipoi][1] = max(bounds[ipoi][1], max_poi)  # limits may lie above the fit range
    kwargs["par_bounds"] = [tuple(b) for b in bounds]

    def hypo(mu):
        obs, exp = pyhf.infer.hypotest(mu, data, model, test_stat=test_stat, return_expected_set=True, **kwargs)
        return float(obs), [float(e) for e in exp]

    def interp(values, cls):
        values, cls = np.asarray(values), np.asarray(cls)
        order = np.argsort(values)
        return toys.interpolate_limit(values[order], cls[order], level)

    def linear(values):
        values = [float(mu) for mu in values if mu > 0]
        results = [hypo(mu) for mu in values]
        # CLs = 1 at mu = 0 by construction
        values = [0.0] + values
        results = [(1.0, [1.0] * 5)] + results
        obs = interp(values, [r[0] for r in results])
        exp = [interp(values, [r[1][k] for r in results]) for k in range(5)]
        return obs, exp

    if isinstance(scan, str) and scan.lower() == "toms748":
        obs, exp = pyhf.infer.intervals.upper_limits.upper_limit(
            data, model, scan=None, level=level, test_stat=test_stat, **kwargs
        )
        return float(obs), [float(e) for e in exp], "toms748"
    if scan:
        return (*linear(np.linspace(float(scan[0]), float(scan[1]), int(scan[2]))), "linear")
    npoints = int(ul_cfg.get("NPoints", 11))
    mu_max = float(ul_cfg.get("Start", 1.0))
    while True:
        obs, exp = hypo(mu_max)
        if (obs < level and max(exp) < level) or mu_max >= max_poi:
            break
        mu_max = min(2.0 * mu_max, max_poi)
    # finer bracketing from above keeps the scan points close to the limits
    obs, exp = linear(np.linspace(0.0, mu_max, npoints))
    return obs, exp, f"scan[0,{mu_max:g},{npoints}]"


def run(ctx: common.WorkspaceContext, general: Dict[str, Any], fit_cfg: Dict[str, Any], outdir: pathlib.Path) -> Dict[str, Any]:
    common.setup_backend(general)
    prep = bkgonly.prepare(ctx, general, fit_cfg)
    model, data = prep["model_fit"], prep["data_fit"]
    poi = model.config.poi_name
    srs = [r for r in ctx.regions_with_role(["SR"]) if r in prep["fit_regions"]]
    log.info(f"discovery fit in {prep['fit_regions']} (POI {poi} = number of signal events in {srs})")
    outdir.mkdir(parents=True, exist_ok=True)

    # free fit and post-fit outputs
    res = bkgonly._fit(model, data, fit_cfg, general)
    common.save_fit_results(res, outdir / "fit_results.json")
    (outdir / "fit_results.txt").write_text(common.print_parameters(res) + "\n")
    # post-fit outputs on the background-only model (dummy signal removed), using the
    # fitted background parameters and their covariance
    dummy = [s for s in model.config.samples if s.startswith("DiscoveryMode_")]
    bkg_spec = dict(pyhf.Workspace(ctx.spec).prune(samples=dummy))
    prep_bkg = bkgonly.prepare(ctx, general, fit_cfg, spec=bkg_spec, poi=None)
    prep_bkg["ranking_model"], prep_bkg["ranking_data"] = model, data
    bkgonly.postfit_outputs(ctx, general, fit_cfg, prep_bkg, res, outdir, ranking_poi=poi)

    kwargs = hypotest_kwargs(model, fit_cfg)
    common.setup_backend(general)
    result: Dict[str, Any] = {"fit_type": "disc", "signal_regions": srs, "poi": poi, "calctype": kwargs["calctype"]}
    if kwargs["calctype"] == "toybased":
        result["ntoys"] = kwargs["ntoys"]
        np.random.seed(int(fit_cfg.get("Seed", 0)))

    # discovery p-value
    p0_obs, p0_exp = pyhf.infer.hypotest(0.0, data, model, test_stat="q0", return_expected=True, **kwargs)
    p0_obs, p0_exp = float(p0_obs), float(p0_exp)
    # HistFitter caps p0 at 0.5 (deficits are not significant)
    result["p0"] = min(p0_obs, 0.5)
    result["Z"] = float(norm.isf(result["p0"]))
    result["p0_expected"] = p0_exp
    result["Z_expected"] = float(norm.isf(min(p0_exp, 0.5)))

    # model-independent upper limits
    if (fit_cfg.get("UpperLimit") or {}).get("Enabled", True):
        if fit_cfg.get("SaveToys") and kwargs["calctype"] == "toybased":
            scan = (fit_cfg.get("UpperLimit") or {}).get("Scan")
            if not scan:
                raise ValueError("SaveToys for a discovery upper limit needs UpperLimit.Scan: [low, high, n]")
            pois = np.linspace(*scan[:2], int(scan[2]))
            seed = int(fit_cfg.get("Seed", 0))
            dist = toys.generate(model, data, pois, ntoys=kwargs["ntoys"], seed=seed, test_stat=fit_cfg.get("TestStat", "qtilde"),
                                 init_pars=kwargs["init_pars"], fixed_params=kwargs["fixed_params"], par_bounds=kwargs["par_bounds"])
            toys.save(dist, outdir / "toys" / f"toys_ul_seed{seed}.npz")
            merged = toys.merge([str(outdir / "toys" / f"toys_ul_seed{seed}.npz")], 1.0 - float(fit_cfg.get("CL", 0.95)))
            obs, exp, method = merged["upper_limit"], merged["expected_upper_limit"], "toys"
        else:
            obs, exp, method = upper_limit(model, data, fit_cfg, kwargs)
        lumi = ctx.meta.get("lumi_fb") or float("nan")
        _, tails = pyhf.infer.hypotest(obs, data, model, test_stat=fit_cfg.get("TestStat", "qtilde"),
                                       return_tail_probs=True, **kwargs)
        result.update(
            {
                "S95_obs": obs,
                "S95_exp": exp[2],
                "S95_exp_band": exp,  # -2, -1, 0, +1, +2 sigma
                "visible_xsec_obs_fb": obs / lumi,
                "visible_xsec_exp_fb": exp[2] / lumi,
                "CLb": float(tails[1]),
                "upper_limit_method": method,
                "lumi_fb": lumi,
            }
        )
    write_json(result, outdir / "discovery.json")
    _write_table(result, outdir / "upper_limit_table")
    log.info(
        f"discovery: p0 = {result['p0']:.3g} (Z = {result['Z']:.2f}), "
        + (f"S95 obs = {result.get('S95_obs', float('nan')):.2f}, exp = {result.get('S95_exp', float('nan')):.2f}")
    )
    return result


def merge_toys(outdir: pathlib.Path, level: float = 0.05) -> Dict[str, Any]:
    """Combines ``toys/toys_ul_seed*.npz`` of several jobs into the upper limits of
    ``discovery.json`` (HistFitter UpperLimitParellel + analyULtoys)."""
    outdir = pathlib.Path(outdir)
    result = read_json(outdir / "discovery.json")
    files = sorted(str(f) for f in (outdir / "toys").glob("toys_ul_seed*.npz"))
    merged = toys.merge(files, level)
    obs, exp = merged["upper_limit"], merged["expected_upper_limit"]
    pois = [p["poi"] for p in merged["points"]]
    clb = float(np.interp(obs, pois, [p["CLb"] for p in merged["points"]]))
    lumi = result.get("lumi_fb") or float("nan")
    result.update({"S95_obs": obs, "S95_exp": exp[2], "S95_exp_band": exp, "visible_xsec_obs_fb": obs / lumi,
                   "visible_xsec_exp_fb": exp[2] / lumi, "CLb": clb, "upper_limit_method": "toys",
                   "ntoys": int(merged["points"][0]["ntoys_b"]), "toy_files": len(files)})
    write_json(result, outdir / "discovery.json")
    _write_table(result, outdir / "upper_limit_table")
    return result


def _write_table(r: Dict[str, Any], path: pathlib.Path) -> None:
    """HistFitter UpperLimitTable layout (txt + tex)."""
    sr = ",".join(r["signal_regions"])
    if "S95_obs" not in r:
        return
    band = r["S95_exp_band"]
    txt = (
        f"{'Signal channel':20s} {'<eps sigma>_obs^95 [fb]':>24s} {'S_obs^95':>10s} {'S_exp^95':>22s} {'CL_B':>6s} {'p(s=0) (Z)':>16s}\n"
        f"{sr:20s} {r['visible_xsec_obs_fb']:24.2f} {r['S95_obs']:10.1f} "
        f"{band[2]:8.1f} +{band[3] - band[2]:.1f} -{band[2] - band[1]:.1f}   {r['CLb']:6.2f} "
        f"{r['p0']:8.2f} ({r['Z']:.2f})\n"
    )
    path.with_suffix(".txt").write_text(txt)
    tex = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\begin{tabular}{lccccc}",
            r"\hline\hline",
            r"{\bf Signal channel} & $\langle\epsilon\sigma\rangle_\mathrm{obs}^{95}$[fb] & $S_\mathrm{obs}^{95}$ & "
            r"$S_\mathrm{exp}^{95}$ & $CL_B$ & $p(s=0)$ ($Z$) \\",
            r"\hline",
            f"{sr.replace('_', ' ')} & ${r['visible_xsec_obs_fb']:.2f}$ & ${r['S95_obs']:.1f}$ & "
            f"${{{band[2]:.1f}}}^{{+{band[3] - band[2]:.1f}}}_{{-{band[2] - band[1]:.1f}}}$ & ${r['CLb']:.2f}$ & "
            f"${r['p0']:.2f}$ (${r['Z']:.2f}$) \\\\",
            r"\hline\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    path.with_suffix(".tex").write_text(tex + "\n")
