"""Post-processing of templates and of the cabinetry-built workspace.

These steps add the HistFitter features that cabinetry does not provide:

* :func:`normalize_systematics` - "Norm" systematics (HistFitter ``overallNormSys`` /
  ``normHistoSys``): for a background normalised with a free parameter in control
  region(s), its variations are rescaled such that the yield in the normalisation
  region(s) is unchanged.  Only the CR->SR extrapolation uncertainty remains.
* :func:`postprocess_workspace` / :func:`postprocess_samples`
    - luminosity uncertainty as a pyhf ``lumi`` modifier (``setNormByTheory``)
    - Poisson-constrained statistical uncertainty of data-driven backgrounds
      (``StatConstraint: Poisson`` -> pyhf ``shapesys``)
    - removal of NaN/inf in modifiers and pruning of negligible modifiers
    - blinding: observed data replaced by the pre-fit background expectation
"""

from __future__ import annotations

import logging
import math
import pathlib
from typing import Any, Dict, List, Optional

import cabinetry
import numpy as np

from fit2l3l.workspace.composer import Registry

log = logging.getLogger(__name__)

#: smallest normsys factor (0 would break the exponential interpolation)
MIN_NORM_FACTOR = 1e-4


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------


def normalize_systematics(config: Dict[str, Any], registry: Registry) -> None:
    """Rescales (post-processed) variation templates to the nominal yield in the
    normalisation regions of free-floating samples."""
    if not registry.normalize_to:
        return
    folder = pathlib.Path(config["General"]["HistogramFolder"])
    regions = {r["Name"]: r for r in config["Regions"]}
    samples = {s["Name"]: s for s in config["Samples"]}
    systs = {s["Name"]: s for s in config.get("Systematics", [])}

    def load(region, sample, syst, template):
        return cabinetry.histo.Histogram.from_config(folder, region, sample, syst, template=template, modified=True)

    for syst_name, per_sample in registry.normalize_to.items():
        syst = systs.get(syst_name)
        if syst is None:
            continue
        for sample_name, norm_regions in per_sample.items():
            sample = samples.get(sample_name)
            if sample is None:
                continue
            affected = [
                r
                for r in config["Regions"]
                if cabinetry.configuration.histogram_is_needed(r, sample, syst, "Up")
            ]
            norm_present = [r for r in norm_regions if r in regions and regions[r] in affected]
            if not norm_present:
                log.warning(f"{syst_name}/{sample_name}: not present in normalisation regions {norm_regions}")
                continue
            nominal_sum = sum(
                float(np.sum(load(regions[r], sample, {}, None).yields)) for r in norm_present
            )
            for template in ("Up", "Down"):
                if syst.get(template, {}).get("Symmetrize", False):
                    continue
                var_sum = sum(
                    float(np.sum(load(regions[r], sample, syst, template).yields)) for r in norm_present
                )
                if var_sum <= 0 or nominal_sum <= 0:
                    continue
                scale = nominal_sum / var_sum
                for region in affected:
                    hist = load(region, sample, syst, template)
                    hist.yields = hist.yields * scale
                    name = cabinetry.histo.name(region, sample, syst, template=template)
                    hist.save(folder / (name + "_modified"))
            log.debug(f"normalised {syst_name} for {sample_name} in {norm_present}")


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------


def _finite(x: float, default: float = 1.0) -> float:
    return x if (x is not None and math.isfinite(x)) else default


def _sanitize_and_prune(sample: Dict[str, Any], pruning: Dict[str, float], channel: str) -> None:
    nominal = np.asarray(sample["data"], dtype=float)
    norm_thr = float(pruning.get("Norm", 0.0))
    shape_thr = float(pruning.get("Shape", 0.0))
    empty = float(np.sum(nominal)) <= 0.0
    kept = []
    for mod in sample["modifiers"]:
        if empty and mod["type"] in ("normsys", "histosys", "shapesys"):
            continue  # sample without events in this channel: systematics are meaningless
        if mod["type"] == "normsys":
            hi, lo = _finite(mod["data"]["hi"]), _finite(mod["data"]["lo"])
            if hi != mod["data"]["hi"] or lo != mod["data"]["lo"]:
                log.warning(f"{channel}/{sample['name']}/{mod['name']}: non-finite normsys set to 1")
            # a variation removing the full yield gives a factor 0, which the
            # exponential interpolation (log) cannot handle
            if min(hi, lo) < MIN_NORM_FACTOR:
                log.debug(f"{channel}/{sample['name']}/{mod['name']}: normsys factor {min(hi, lo):.2g} set to {MIN_NORM_FACTOR}")
                hi, lo = max(hi, MIN_NORM_FACTOR), max(lo, MIN_NORM_FACTOR)
            mod["data"] = {"hi": hi, "lo": lo}
            if max(abs(hi - 1.0), abs(lo - 1.0)) <= max(norm_thr, 1e-9):
                continue
        elif mod["type"] == "histosys":
            hi = np.asarray(mod["data"]["hi_data"], dtype=float)
            lo = np.asarray(mod["data"]["lo_data"], dtype=float)
            hi = np.where(np.isfinite(hi), hi, nominal)
            lo = np.where(np.isfinite(lo), lo, nominal)
            mod["data"] = {"hi_data": hi.tolist(), "lo_data": lo.tolist()}
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.nan_to_num(np.maximum(np.abs(hi - nominal), np.abs(lo - nominal)) / nominal)
            if np.max(rel, initial=0.0) <= max(shape_thr, 1e-9):
                continue
        elif mod["type"] == "staterror":
            mod["data"] = [x if math.isfinite(x) else 0.0 for x in mod["data"]]
        kept.append(mod)
    sample["modifiers"] = kept


def _add_lumi(sample: Dict[str, Any]) -> None:
    if not any(m["type"] == "lumi" for m in sample["modifiers"]):
        sample["modifiers"].append({"name": "lumi", "type": "lumi", "data": None})


def _poisson_stat(sample: Dict[str, Any], registry: Registry) -> None:
    """Replaces the stat_<sample>_<region> normsys/histosys by a shapesys."""
    new = []
    done = set()
    for mod in sample["modifiers"]:
        info = registry.poisson_stat.get(mod["name"], {}).get(sample["name"])
        if info is None:
            new.append(mod)
            continue
        if mod["name"] not in done:
            new.append({"name": mod["name"], "type": "shapesys", "data": [float(x) for x in info]})
            done.add(mod["name"])
    sample["modifiers"] = new


def postprocess_samples(
    channel: str, samples: List[Dict[str, Any]], registry: Registry, pruning: Optional[Dict[str, float]] = None
) -> None:
    """Applies lumi / Poisson / sanitising / pruning to the samples of a channel."""
    pruning = pruning or {}
    for sample in samples:
        if registry.stat_constraint == "Poisson":
            _poisson_stat(sample, registry)
        _sanitize_and_prune(sample, pruning, channel)
        if sample["name"] in registry.lumi_samples:
            _add_lumi(sample)


def lumi_parameter(registry: Registry) -> Dict[str, Any]:
    sigma = registry.lumi_uncertainty
    return {
        "name": "lumi",
        "auxdata": [1.0],
        "sigmas": [sigma],
        "bounds": [[max(0.0, 1.0 - 5.0 * sigma), 1.0 + 5.0 * sigma]],
        "inits": [1.0],
    }


def blind_observations(ws: Dict[str, Any], regions: List[str], exclude_samples: List[str]) -> None:
    """Replaces the data in ``regions`` by the nominal background expectation."""
    for channel in ws["channels"]:
        if channel["name"] not in regions:
            continue
        total = np.zeros(len(channel["samples"][0]["data"]))
        for sample in channel["samples"]:
            if sample["name"] in exclude_samples:
                continue
            total += np.asarray(sample["data"], dtype=float)
        for obs in ws["observations"]:
            if obs["name"] == channel["name"]:
                obs["data"] = total.tolist()
        log.info(f"blinded {channel['name']}: data replaced by the background expectation {total.round(3).tolist()}")


def postprocess_workspace(
    ws: Dict[str, Any], registry: Registry, pruning: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    for channel in ws["channels"]:
        postprocess_samples(channel["name"], channel["samples"], registry, pruning)
    if registry.lumi_samples:
        params = ws["measurements"][0]["config"]["parameters"]
        if not any(p["name"] == "lumi" for p in params):
            params.append(lumi_parameter(registry))
    if registry.blinded_regions:
        blind_observations(ws, registry.blinded_regions, registry.signal_samples + _discovery(ws))
    return ws


def _discovery(ws: Dict[str, Any]) -> List[str]:
    return [s["name"] for c in ws["channels"] for s in c["samples"] if s["name"].startswith("DiscoveryMode_")]
