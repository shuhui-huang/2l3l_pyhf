"""Runs the fits described by a fit YAML on a workspace (ws.json)."""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Any, Dict, List, Optional, Sequence

from fit2l3l.fitting import common
from fit2l3l.utils import as_list, dump_yaml, load_yaml

log = logging.getLogger(__name__)


def load_fit_config(config: Any, *, workspace: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_yaml(config) if not isinstance(config, dict) else dict(config)
    unknown = set(cfg) - {"General", "Fits", "Contours"}
    if unknown:
        raise ValueError(f"unknown top-level keys in fit YAML: {sorted(unknown)}")
    general = common.merge_defaults(cfg.get("General", {}), common.GENERAL_DEFAULTS)
    if workspace:
        general["Workspace"] = workspace
    if not general.get("Workspace"):
        raise ValueError("General.Workspace (directory with ws.json) is required (or --workspace)")
    fits = [common.merge_defaults(f, common.FIT_DEFAULTS) for f in as_list(cfg.get("Fits")) or [{}]]
    for i, fit in enumerate(fits):
        fit["Name"] = fit["Name"] or f"fit{i}"
    return {"General": general, "Fits": fits, "Contours": cfg.get("Contours")}


def run_fits(
    config: Any,
    *,
    workspace: Optional[str] = None,
    only: Optional[Sequence[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    merge: bool = True,
) -> Dict[str, Any]:
    cfg = load_fit_config(config, workspace=workspace)
    general = cfg["General"]
    ctx = common.load_context(general["Workspace"])
    base_out = pathlib.Path(general["OutputDir"]) if general.get("OutputDir") else ctx.directory / "fits"
    results = {}
    for fit_cfg in cfg["Fits"]:
        if only and fit_cfg["Name"] not in only:
            continue
        fit_cfg = dict(fit_cfg)
        fit_cfg.update(overrides or {})
        fit_type = fit_cfg.get("Type") or ctx.meta.get("fit_type", "bkg")
        outdir = base_out / fit_cfg["Name"]
        outdir.mkdir(parents=True, exist_ok=True)
        dump_yaml({"General": general, "Fit": fit_cfg}, outdir / "fit_config_used.yaml")
        t0 = time.time()
        log.info(f"=== fit '{fit_cfg['Name']}' ({fit_type}) on {ctx.directory / 'ws.json'} -> {outdir}")
        if fit_type == "bkg":
            from fit2l3l.fitting import bkgonly

            results[fit_cfg["Name"]] = bkgonly.run(ctx, general, fit_cfg, outdir)
        elif fit_type == "disc":
            from fit2l3l.fitting import discovery

            results[fit_cfg["Name"]] = discovery.run(ctx, general, fit_cfg, outdir)
        elif fit_type == "excl":
            from fit2l3l.fitting import exclusion

            results[fit_cfg["Name"]] = exclusion.run(
                ctx, general, fit_cfg, outdir, contours=cfg.get("Contours"), do_merge=merge
            )
        else:
            raise ValueError(f"unknown fit type '{fit_type}'")
        log.info(f"=== fit '{fit_cfg['Name']}' done in {time.time() - t0:.1f} s")
    return results
