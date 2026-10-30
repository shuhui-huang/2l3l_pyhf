"""Shared helpers of the pyhf fit runner."""

from __future__ import annotations

import copy
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cabinetry
import numpy as np
import pyhf

from fit2l3l.utils import as_list, read_json, write_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# fit configuration (YAML) defaults
# ---------------------------------------------------------------------------

GENERAL_DEFAULTS = {
    "Workspace": None,  # directory with ws.json + meta.json, or path to ws.json
    "OutputDir": None,  # default: <workspace dir>/fits
    "Backend": "numpy",
    # optimizer for hypothesis tests / limits (pyhf.infer): scipy is several times
    # faster than minuit with the numpy backend; maximum-likelihood fits with
    # parameter uncertainties always use minuit (through cabinetry)
    "Optimizer": "scipy",  # scipy | minuit
    "Strategy": None,
    "Tolerance": None,
    "MaxIter": None,
    # pyhf interpolation codes, e.g. {normsys: code1, histosys: code0}
    # (pyhf defaults: normsys code4, histosys code4p)
    "InterpolationCodes": {},
    "ATLASLabel": "Internal",
    "PlotFormats": ["pdf", "png"],
}

FIT_DEFAULTS = {
    "Name": None,
    "Type": None,  # bkg | disc | excl (default: workspace fit type)
    "FitRegions": None,  # default: regions flagged "fit" in meta.json
    "ValidationRegions": None,  # default: remaining regions
    "Blind": [],  # roles or region names whose data are replaced by the bkg expectation
    "Asimov": False,  # fit the (background-only) Asimov dataset
    "Fixed": {},  # {parameter: value}
    "Init": {},  # {parameter: value}
    "Bounds": {},  # {parameter: [low, high]}
    "Minos": [],
    "Plots": ["prefit", "postfit", "pulls", "correlation", "summary"],
    "Tables": ["yields", "systematics"],
    "Ranking": {"Enabled": False, "MaxPars": 20},
    "SystematicTable": {"Parameters": True, "Groups": True, "PerSample": False},
    "LogY": None,
    # hypothesis tests / limits
    "Calculator": "asymptotics",  # asymptotics | toybased
    "TestStat": "qtilde",
    "NToys": 1000,
    "Seed": 0,
    "SaveToys": False,  # save toy test-statistic distributions (merge later)
    "CL": 0.95,
    # Scan: null (automatic range, NPoints), [low, high, n] or "toms748" (root finding)
    "UpperLimit": {"Enabled": True, "Scan": None, "NPoints": 11},
    "TheoryVariations": True,  # observed CLs at mu = 1 +- dsigma (signal xsec +-1 sigma)
    "Points": "all",
    "PointFits": False,  # save the free fit for every signal point
    "Jobs": 1,  # parallel processes for the signal points
}


def merge_defaults(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(defaults)
    for key, value in (cfg or {}).items():
        if key not in defaults:
            raise ValueError(f"unknown fit configuration key '{key}', allowed: {sorted(defaults)}")
        if isinstance(defaults[key], dict) and isinstance(value, dict) and key not in ("Fixed", "Init", "Bounds", "InterpolationCodes"):
            merged = dict(defaults[key])
            merged.update(value)
            out[key] = merged
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# workspace context
# ---------------------------------------------------------------------------


@dataclass
class WorkspaceContext:
    directory: pathlib.Path
    spec: Dict[str, Any]
    meta: Dict[str, Any]
    patchset: Optional[pyhf.PatchSet] = None
    cabinetry_config: Optional[Dict[str, Any]] = None

    @property
    def workspace(self) -> pyhf.Workspace:
        return pyhf.Workspace(self.spec)

    def region(self, name: str) -> Dict[str, Any]:
        for reg in self.meta.get("regions", []):
            if reg["name"] == name:
                return reg
        return {"name": name, "role": "?", "fit": True, "variable": "cuts", "label": name, "blinded": False}

    def regions_with_role(self, roles: Sequence[str]) -> List[str]:
        return [r["name"] for r in self.meta.get("regions", []) if r["role"] in roles]


def load_context(path: Any) -> WorkspaceContext:
    path = pathlib.Path(path)
    ws_file = path if path.suffix == ".json" else path / "ws.json"
    directory = ws_file.parent
    if not ws_file.exists():
        raise FileNotFoundError(f"workspace {ws_file} not found - build it first (fit2l3l build ...)")
    spec = json.loads(ws_file.read_text())
    meta_file = directory / "meta.json"
    meta = read_json(meta_file) if meta_file.exists() else _meta_from_spec(spec)
    patchset = None
    if (directory / "patchset.json").exists():
        patchset = pyhf.PatchSet(read_json(directory / "patchset.json"))
    cab = None
    if (directory / "cabinetry_config.yaml").exists():
        import yaml

        cab = yaml.safe_load((directory / "cabinetry_config.yaml").read_text())
    return WorkspaceContext(directory=directory, spec=spec, meta=meta, patchset=patchset, cabinetry_config=cab)


def _meta_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal metadata for workspaces not built by fit2l3l (e.g. from HEPData)."""
    poi = spec["measurements"][0]["config"].get("poi", "")
    return {
        "name": "workspace",
        "fit_type": "excl" if poi else "bkg",
        "poi": poi or "mu_SIG",
        "lumi_fb": None,
        "regions": [
            {"name": c["name"], "role": "SR", "fit": True, "variable": "obs", "label": c["name"], "blinded": False,
             "bins": list(range(len(c["samples"][0]["data"]) + 1))}
            for c in spec["channels"]
        ],
        "samples": {},
        "systematic_groups": {},
    }


# ---------------------------------------------------------------------------
# model / data
# ---------------------------------------------------------------------------


def setup_backend(general: Dict[str, Any], *, verbose: int = 0) -> None:
    optimizer = general.get("Optimizer", "minuit")
    kwargs = {}
    if optimizer == "minuit":
        if general.get("Strategy") is not None:
            kwargs["strategy"] = general["Strategy"]
        if general.get("Tolerance") is not None:
            kwargs["tolerance"] = general["Tolerance"]
        opt = pyhf.optimize.minuit_optimizer(verbose=verbose, **kwargs)
    else:
        opt = pyhf.optimize.scipy_optimizer(verbose=verbose)
    pyhf.set_backend(general.get("Backend", "numpy"), opt)


def modifier_settings(general: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    codes = general.get("InterpolationCodes") or {}
    if not codes:
        return None
    return {k: {"interpcode": v} for k, v in codes.items()}


def build_model(
    spec: Dict[str, Any],
    general: Dict[str, Any],
    *,
    channels: Optional[Sequence[str]] = None,
    poi: Optional[str] = "default",
) -> Tuple[pyhf.Workspace, pyhf.pdf.Model, List[float]]:
    """Returns (workspace, model, data) restricted to ``channels``."""
    ws = pyhf.Workspace(spec)
    if channels is not None:
        drop = [c for c in ws.channels if c not in channels]
        if drop:
            ws = ws.prune(channels=drop)
    kwargs = {}
    settings = modifier_settings(general)
    if settings:
        kwargs["modifier_settings"] = settings
    if poi != "default":
        kwargs["poi_name"] = poi
    elif not ws.get_measurement()["config"].get("poi"):
        kwargs["poi_name"] = None
    model = ws.model(**kwargs)
    data = ws.data(model)
    return ws, model, data


def parameter_settings(model: pyhf.pdf.Model, fit_cfg: Dict[str, Any]):
    """init_pars / fix_pars / par_bounds lists from the fit configuration."""
    init = model.config.suggested_init()
    fixed = model.config.suggested_fixed()
    bounds = [list(b) for b in model.config.suggested_bounds()]
    names = model.config.par_names
    for key, apply in (("Init", "init"), ("Fixed", "fixed"), ("Bounds", "bounds")):
        for par, value in (fit_cfg.get(key) or {}).items():
            idx = [i for i, n in enumerate(names) if n == par or n.startswith(par + "[")]
            if not idx:
                log.warning(f"{key}: parameter '{par}' not in the model")
                continue
            for i in idx:
                if apply == "init":
                    init[i] = float(value)
                elif apply == "fixed":
                    init[i] = float(value)
                    fixed[i] = True
                else:
                    bounds[i] = [float(value[0]), float(value[1])]
    return init, fixed, [tuple(b) for b in bounds]


def asimov_background(model: pyhf.pdf.Model, init: Optional[List[float]] = None) -> np.ndarray:
    """Expected main-model data (no auxdata) for the nominal parameters, POI at 0."""
    pars = np.asarray(init if init is not None else model.config.suggested_init(), dtype=float)
    if model.config.poi_index is not None:
        pars[model.config.poi_index] = 0.0
    return np.asarray(model.expected_data(pars, include_auxdata=False))


def blind_data(
    model: pyhf.pdf.Model, data: Sequence[float], channels: Sequence[str]
) -> List[float]:
    """Replaces the observed data of ``channels`` by the background expectation."""
    if not channels:
        return list(data)
    data = np.asarray(data, dtype=float).copy()
    expected = asimov_background(model)
    for channel in channels:
        if channel not in model.config.channels:
            continue
        sl = model.config.channel_slices[channel]
        data[sl] = expected[sl]
    return data.tolist()


def resolve_blinded(ctx: WorkspaceContext, setting: Any) -> List[str]:
    out = []
    for item in as_list(setting):
        up = str(item).upper()
        if up == "ALL":
            out += [r["name"] for r in ctx.meta["regions"]]
        elif up in ("SR", "CR", "VR"):
            out += ctx.regions_with_role([up])
        else:
            out.append(item)
    return out


def fit_and_validation_regions(ctx: WorkspaceContext, fit_cfg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    all_regions = [c["name"] for c in ctx.spec["channels"]]
    if fit_cfg.get("FitRegions"):
        fit = resolve_blinded(ctx, fit_cfg["FitRegions"])
    else:
        fit = [r["name"] for r in ctx.meta["regions"] if r.get("fit", True) and r["name"] in all_regions]
    if fit_cfg.get("ValidationRegions"):
        val = resolve_blinded(ctx, fit_cfg["ValidationRegions"])
    else:
        val = [r for r in all_regions if r not in fit]
    unknown = [r for r in fit + val if r not in all_regions]
    if unknown:
        raise ValueError(f"regions {unknown} are not in the workspace (channels: {all_regions})")
    return fit, val


# ---------------------------------------------------------------------------
# fit results I/O
# ---------------------------------------------------------------------------


def fit_results_to_dict(res: cabinetry.fit.FitResults) -> Dict[str, Any]:
    return {
        "labels": list(res.labels),
        "bestfit": np.asarray(res.bestfit).tolist(),
        "uncertainty": np.asarray(res.uncertainty).tolist(),
        "corr_mat": np.asarray(res.corr_mat).tolist(),
        "best_twice_nll": float(res.best_twice_nll),
        "goodness_of_fit": float(res.goodness_of_fit),
        "minos_uncertainty": {k: list(v) for k, v in (res.minos_uncertainty or {}).items()},
    }


def fit_results_from_dict(d: Dict[str, Any]) -> cabinetry.fit.FitResults:
    return cabinetry.fit.FitResults(
        np.asarray(d["bestfit"]),
        np.asarray(d["uncertainty"]),
        list(d["labels"]),
        np.asarray(d["corr_mat"]),
        float(d["best_twice_nll"]),
        float(d.get("goodness_of_fit", -1)),
        {k: tuple(v) for k, v in d.get("minos_uncertainty", {}).items()},
    )


def save_fit_results(res: cabinetry.fit.FitResults, path: pathlib.Path) -> None:
    write_json(fit_results_to_dict(res), path)


def print_parameters(res: cabinetry.fit.FitResults, only: Optional[Sequence[str]] = None) -> str:
    lines = []
    for label, value, unc in zip(res.labels, res.bestfit, res.uncertainty):
        if only and not any(label.startswith(o) for o in only):
            continue
        minos = res.minos_uncertainty.get(label) if res.minos_uncertainty else None
        extra = f"  (MINOS {minos[0]:+.4f} / {minos[1]:+.4f})" if minos else ""
        lines.append(f"{label:45s} = {value:9.4f} +/- {unc:.4f}{extra}")
    return "\n".join(lines)


def parameter_group(name: str, groups: Dict[str, str]) -> str:
    """Group of a parameter (for systematic tables)."""
    base = name.split("[")[0]
    if base in groups:
        return groups[base]
    if base.startswith("staterror_"):
        return "MC stat."
    if base.startswith("stat_"):
        return "DD stat."
    if base.startswith("syst_"):
        return "DD syst."
    if base == "lumi":
        return "Luminosity"
    if base.startswith("mu_"):
        return "Norm. factors"
    if base.startswith("theoryUncert"):
        return "TH"
    return "Other"
