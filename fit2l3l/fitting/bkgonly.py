"""Background-only fit (HistFitter ``-F bkg``) and the post-fit outputs shared by all fits.

* maximum-likelihood fit in the fit regions (CRs; SRs/VRs as configured)
* extrapolation to the validation regions using the fitted parameters and their
  covariance (parameters only present in VRs keep their pre-fit values)
* pre-/post-fit plots per region, summary plot, pulls, correlations, ranking
* yield tables and systematic breakdown tables (HistFitter YieldsTable/SysTable)
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Sequence

import cabinetry
import numpy as np

from fit2l3l.fitting import common, tables
from fit2l3l.plotting import datamc
from fit2l3l.utils import write_json

log = logging.getLogger(__name__)


def _fit(model, data, fit_cfg, general):
    init, fixed, bounds = common.parameter_settings(model, fit_cfg)
    minos = fit_cfg.get("Minos") or None
    return cabinetry.fit.fit(
        model,
        data,
        minos=minos,
        goodness_of_fit=True,
        init_pars=init,
        fix_pars=fixed,
        par_bounds=bounds,
        strategy=general.get("Strategy"),
        maxiter=general.get("MaxIter"),
        tolerance=general.get("Tolerance"),
    )


def prepare(ctx: common.WorkspaceContext, general: Dict[str, Any], fit_cfg: Dict[str, Any], *, spec=None, poi="default"):
    """Builds fit and full models and the (possibly blinded / Asimov) data."""
    spec = spec if spec is not None else ctx.spec
    fit_regions, val_regions = common.fit_and_validation_regions(ctx, fit_cfg)
    _, model_full, data_full = common.build_model(spec, general, poi=poi)
    _, model_fit, data_fit = common.build_model(spec, general, channels=fit_regions, poi=poi)
    blinded_fit_time = common.resolve_blinded(ctx, fit_cfg.get("Blind"))
    if fit_cfg.get("Asimov"):
        init, _, _ = common.parameter_settings(model_fit, fit_cfg)
        data_fit = list(common.asimov_background(model_fit, init)) + list(model_fit.config.auxdata)
        data_full = list(common.asimov_background(model_full)) + list(model_full.config.auxdata)
        blinded_fit_time = [c for c in model_full.config.channels]
    else:
        data_fit = common.blind_data(model_fit, data_fit, blinded_fit_time)
        data_full = common.blind_data(model_full, data_full, blinded_fit_time)
    blinded = sorted(set(blinded_fit_time) | set(ctx.meta.get("blinded_regions", [])))
    return dict(
        fit_regions=fit_regions,
        val_regions=val_regions,
        model_fit=model_fit,
        data_fit=data_fit,
        model_full=model_full,
        data_full=data_full,
        blinded=blinded,
    )


def postfit_outputs(
    ctx: common.WorkspaceContext,
    general: Dict[str, Any],
    fit_cfg: Dict[str, Any],
    prep: Dict[str, Any],
    res: cabinetry.fit.FitResults,
    outdir: pathlib.Path,
    *,
    exclude_samples: Sequence[str] = (),
    ranking_poi: Optional[str] = None,
) -> Dict[str, Any]:
    model_full, data_full = prep["model_full"], prep["data_full"]
    res_full = cabinetry.model_utils.match_fit_results(model_full, res)
    prefit = cabinetry.model_utils.prediction(model_full)
    postfit = cabinetry.model_utils.prediction(model_full, fit_results=res_full)
    data = datamc.data_per_channel(model_full, data_full)
    plots = set(fit_cfg.get("Plots") or [])
    tabs = set(fit_cfg.get("Tables") or [])
    formats = general.get("PlotFormats", ["pdf", "png"])
    atlas = general.get("ATLASLabel", "Internal")
    lumi = ctx.meta.get("lumi_fb")
    samples_meta = ctx.meta.get("samples", {})
    blinded = prep["blinded"]
    outputs: Dict[str, Any] = {}
    channels = list(model_full.config.channels)
    # order of the workspace YAML: CRs, VRs, SRs
    meta_order = [r["name"] for role in ("CR", "VR", "SR") for r in ctx.meta.get("regions", []) if r["role"] == role]
    ordered = [c for c in meta_order if c in channels] + [c for c in channels if c not in meta_order]

    for label, pred, flag in (("prefit", prefit, "prefit"), ("postfit", postfit, "postfit")):
        if flag not in plots:
            continue
        for channel in ordered:
            datamc.plot_region(
                pred, data, channel, ctx.region(channel), samples_meta, outdir / "plots" / f"{channel}_{label}",
                blinded=channel in blinded, atlas=atlas, lumi_fb=lumi, log_y=fit_cfg.get("LogY"), formats=formats,
            )
    if "summary" in plots:
        for label, pred in (("prefit", prefit), ("postfit", postfit)):
            datamc.plot_summary(pred, data, ordered, samples_meta, outdir / "plots" / f"summary_{label}",
                                blinded=blinded, atlas=atlas, lumi_fb=lumi, formats=formats)
    if "pulls" in plots:
        # MC-stat gammas, free normalisation factors and fixed parameters are not shown
        exclude = [
            l for l, u in zip(res.labels, res.uncertainty)
            if l.startswith(("staterror_", "mu_")) or u == 0.0
        ]
        try:
            cabinetry.visualize.pulls(res, figure_folder=outdir / "plots", exclude=exclude, close_figure=True)
        except Exception as exc:  # pragma: no cover - plotting must not stop the fit
            log.warning(f"pull plot failed: {exc}")
    if "correlation" in plots:
        cabinetry.visualize.correlation_matrix(res, figure_folder=outdir / "plots", pruning_threshold=0.1, close_figure=True)
    if "modifier_grid" in plots:
        cabinetry.visualize.modifier_grid(prep["model_fit"], figure_folder=outdir / "plots", close_figure=True)

    rank_cfg = fit_cfg.get("Ranking") or {}
    if rank_cfg.get("Enabled"):
        poi = rank_cfg.get("POI") or ranking_poi or next((l for l in res.labels if l.startswith("mu_")), None)
        if poi is None:
            log.warning("ranking needs a POI or a free normalisation factor")
        else:
            rmodel = prep.get("ranking_model", prep["model_fit"])
            rdata = prep.get("ranking_data", prep["data_fit"])
            init, fixed, bounds = common.parameter_settings(rmodel, fit_cfg)
            ranking = cabinetry.fit.ranking(rmodel, rdata, fit_results=res, poi_name=poi,
                                            init_pars=init, fix_pars=fixed, par_bounds=bounds)
            cabinetry.visualize.ranking(ranking, figure_folder=outdir / "plots", max_pars=rank_cfg.get("MaxPars", 20))
            write_json({"poi": poi, "labels": ranking.labels, "bestfit": ranking.bestfit, "uncertainty": ranking.uncertainty,
                        "prefit_up": ranking.prefit_up, "prefit_down": ranking.prefit_down,
                        "postfit_up": ranking.postfit_up, "postfit_down": ranking.postfit_down}, outdir / "ranking.json")

    if "yields" in tabs:
        for name, chans in (("fit_regions", prep["fit_regions"]), ("validation_regions", prep["val_regions"])):
            if not chans:
                continue
            table = tables.yields_table(prefit, postfit, data, chans, blinded=blinded, exclude_samples=exclude_samples)
            outputs[f"yields_{name}"] = tables.write_yields_table(
                table, outdir / "tables" / f"yields_{name}", samples_meta=samples_meta,
                title=f"{ctx.meta.get('name', '')}: yields in the {name.replace('_', ' ')}")
            cabinetry.tabulate.yields(postfit, data_full, channels=chans, per_bin=True, per_channel=False,
                                      table_folder=outdir / "tables" / f"per_bin_{name}", table_format="simple")
    if "systematics" in tabs:
        st = fit_cfg.get("SystematicTable") or {}
        groups = ctx.meta.get("systematic_groups", {})
        breakdown = tables.uncertainty_breakdown(
            model_full, res_full.bestfit, res_full.uncertainty, res_full.corr_mat, ordered, groups,
            exclude_samples=exclude_samples,
        )
        outputs["systematics"] = tables.write_systematics_table(
            breakdown, outdir / "tables" / "systematics_postfit", title=f"{ctx.meta.get('name', '')}: post-fit uncertainties",
            parameters=st.get("Parameters", True), groups=st.get("Groups", True))
        tables.plot_grouped_uncertainties(breakdown, outdir / "plots" / "uncertainties_grouped_postfit",
                                          atlas=atlas, lumi_fb=lumi, formats=formats)
        asimov = cabinetry.model_utils.asimov_parameters(model_full)
        prefit_unc = cabinetry.model_utils.prefit_uncertainties(model_full)
        breakdown_pre = tables.uncertainty_breakdown(
            model_full, asimov, prefit_unc, np.eye(len(asimov)), ordered, groups, exclude_samples=exclude_samples)
        tables.write_systematics_table(
            breakdown_pre, outdir / "tables" / "systematics_prefit", title=f"{ctx.meta.get('name', '')}: pre-fit uncertainties",
            parameters=st.get("Parameters", True), groups=st.get("Groups", True))
    return outputs


def run(ctx: common.WorkspaceContext, general: Dict[str, Any], fit_cfg: Dict[str, Any], outdir: pathlib.Path) -> Dict[str, Any]:
    common.setup_backend(general)
    prep = prepare(ctx, general, fit_cfg, poi=None)
    log.info(f"background-only fit in {prep['fit_regions']}, validation regions: {prep['val_regions']}")
    res = _fit(prep["model_fit"], prep["data_fit"], fit_cfg, general)
    outdir.mkdir(parents=True, exist_ok=True)
    common.save_fit_results(res, outdir / "fit_results.json")
    text = common.print_parameters(res)
    (outdir / "fit_results.txt").write_text(text + f"\n\n-2lnL = {res.best_twice_nll:.4f}, GoF p-value = {res.goodness_of_fit:.4f}\n")
    log.info("fitted normalisation factors:\n" + common.print_parameters(res, only=["mu_"]))
    outputs = postfit_outputs(ctx, general, fit_cfg, prep, res, outdir)
    summary = {
        "fit_type": "bkg",
        "fit_regions": prep["fit_regions"],
        "validation_regions": prep["val_regions"],
        "normfactors": {l: [float(v), float(u)] for l, v, u in zip(res.labels, res.bestfit, res.uncertainty) if l.startswith("mu_")},
        "twice_nll": float(res.best_twice_nll),
        "gof_pvalue": float(res.goodness_of_fit),
    }
    write_json(summary, outdir / "summary.json")
    return summary
