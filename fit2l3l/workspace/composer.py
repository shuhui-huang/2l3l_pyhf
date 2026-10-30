"""Turns a top-level workspace YAML + python analysis definitions into a cabinetry config.

The workspace YAML only *selects* what enters the workspace (regions with their
role, samples, systematic groups and a few options).  All details (cuts, binning,
weights, trees, systematic definitions, hard-coded yields) are taken from the
python analysis module (``fit2l3l.analysis.<name>``).

The result of :func:`compose` is a :class:`Composition` holding

* ``config``   - a valid cabinetry configuration (dict); dumped to
  ``<output>/cabinetry_config.yaml`` for inspection
* ``registry`` - information the template builder needs beyond what cabinetry
  config can express (data-driven yields, region/bin dependent theory
  uncertainties, overflow settings, normalisation-to-CR of systematics, ...)
* ``meta``     - metadata written next to ``ws.json`` and used by the fit runner
  (region roles, colours, signal points, luminosity, ...)

Conventions for ``Up``/``Down`` template overrides produced here (resolved by
:mod:`fit2l3l.workspace.templates`):

* ``Tree: "{prefix}_JET_X__1up"``     -> ``{prefix}`` is the sample's tree prefix
* ``Weight: "{weight}*wpdf_up"``       -> ``{weight}`` is the sample's nominal weight
* ``Weight: "{weight:wmu_nom->wmu_stat_up}"`` -> nominal weight with one factor replaced
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fit2l3l.analysis.base import (
    CUTS,
    DATA,
    DATADRIVEN,
    DISCOVERY,
    MC,
    SIGNAL,
    Analysis,
    NormSyst,
    Sample,
    SignalGrid,
    SignalPoint,
    TheorySyst,
    TreeSyst,
    WeightSyst,
)
from fit2l3l.utils import as_list, load_analysis, unique
from fit2l3l.workspace import expressions

log = logging.getLogger(__name__)

FIT_TYPES = ("bkg", "excl", "disc")
ROLES = ("SR", "CR", "VR")

DEFAULT_GENERAL = {
    "Analysis": "ss3l",
    "FitType": "bkg",
    "Lumi": 138965,
    "LumiUncertainty": 0.017,
    "InputPath": "InputTrees",
    "InputType": "ntuple",
    "OutputDir": "results",
    "Systematics": "NONE",
    "Blind": [],
    "StatConstraint": "Gaussian",
    "BinnedTheoryUncertainty": False,
    "POI": "mu_SIG",
    "POIBounds": None,  # default: [0, 10] (excl), [0, 100] (disc)
    "NormalizeSystematicsInCR": True,
    "MissingTree": "nominal",  # nominal | error : missing systematic trees
    "Measurement": "NormalMeasurement",
    "Options": {},
    "HEPDataRecord": "ins0000000",  # inspire id for the patchset metadata
    # missing hard-coded (data-driven) yields for a region: error | zero (HistFitter: zero)
    "MissingYields": "error",
    "Description": "",
}


# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------


@dataclass
class RegionEntry:
    name: str
    role: str  # declared role in the YAML: SR / CR / VR
    variable: str = CUTS
    overflow: Optional[bool] = None
    underflow: Optional[bool] = None
    fit: bool = True  # included in the likelihood fit (False: validation region)
    blinded: bool = False


@dataclass
class CustomTemplate:
    """How to build a template that is not a plain ntuple projection."""

    kind: str  # "yields" | "scale" | "ntuple"
    # yields: callable(region_name) -> (values, variances) ; for systematics: variation
    yields: Optional[Any] = None
    # scale: callable(region_name, nbins) -> factors (array) applied to the nominal
    factors: Optional[Any] = None
    # ntuple: callable(region_name) -> weight expression (string), may use {weight}
    weight: Optional[Any] = None


@dataclass
class Registry:
    """Extra information for the template builder / post-processing."""

    analysis: Analysis
    input_type: str
    input_path: str
    lumi: float
    missing_tree: str = "nominal"
    regions: Dict[str, RegionEntry] = field(default_factory=dict)
    samples: Dict[str, Sample] = field(default_factory=dict)
    sample_weights: Dict[str, List[str]] = field(default_factory=dict)
    # (sample, systematic-name or "" for nominal, template or None) -> CustomTemplate
    custom: Dict[Tuple[str, str, Optional[str]], CustomTemplate] = field(default_factory=dict)
    # systematic name -> {sample: [regions]} : HistFitter "Norm" systematics
    normalize_to: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    lumi_samples: List[str] = field(default_factory=list)
    lumi_uncertainty: float = 0.017
    # systematic name -> per-region absolute stat uncertainties (for Poisson conversion)
    poisson_stat: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    stat_constraint: str = "Gaussian"
    blinded_regions: List[str] = field(default_factory=list)
    signal_samples: List[str] = field(default_factory=list)


@dataclass
class Composition:
    config: Dict[str, Any]
    registry: Registry
    meta: Dict[str, Any]
    signal_points: List[SignalPoint] = field(default_factory=list)
    grid: Optional[SignalGrid] = None

    def config_for(self, signal: Optional[str]) -> Dict[str, Any]:
        """cabinetry config restricted to one signal point (or none: bkg-only)."""
        cfg = copy.deepcopy(self.config)
        drop = {s for s in self.registry.signal_samples if s != signal}
        cfg["Samples"] = [s for s in cfg["Samples"] if s["Name"] not in drop]
        for key in ("NormFactors", "Systematics"):
            new = []
            for item in cfg.get(key, []):
                if "Samples" in item:
                    samples = [s for s in as_list(item["Samples"]) if s not in drop]
                    if not samples:
                        if key == "NormFactors":
                            # keep the parameter definition (POI bounds) without samples
                            item = dict(item, Samples=["signal"])
                            new.append(item)
                        continue
                    item = dict(item, Samples=samples)
                new.append(item)
            cfg[key] = new
        if signal is None and self.meta["fit_type"] == "excl":
            cfg["General"]["POI"] = ""
        return cfg


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def _parse_regions(block: Dict[str, Any]) -> List[RegionEntry]:
    entries: List[RegionEntry] = []
    if not block:
        raise ValueError("the workspace YAML needs a 'Regions' block with SR/CR/VR lists")
    for role, items in block.items():
        role_u = role.upper()
        if role_u not in ROLES:
            raise ValueError(f"unknown region role '{role}', use one of {ROLES}")
        for item in as_list(items):
            if isinstance(item, str):
                name, variable = (item.split(":", 1) + [CUTS])[:2] if ":" in item else (item, CUTS)
                entries.append(RegionEntry(name=name.strip(), role=role_u, variable=variable.strip()))
            elif isinstance(item, dict):
                unknown = set(item) - {"Name", "Variable", "Overflow", "Underflow"}
                if unknown:
                    raise ValueError(f"unknown keys {unknown} in region entry {item}")
                entries.append(
                    RegionEntry(
                        name=item["Name"],
                        role=role_u,
                        variable=item.get("Variable", CUTS),
                        overflow=item.get("Overflow"),
                        underflow=item.get("Underflow"),
                    )
                )
            else:
                raise ValueError(f"cannot parse region entry {item!r}")
    names = [e.name for e in entries]
    dupl = {n for n in names if names.count(n) > 1}
    if dupl:
        raise ValueError(f"regions listed more than once: {sorted(dupl)}")
    return entries


def _assign_fit_roles(entries: List[RegionEntry], fit_type: str) -> None:
    """HistFitter conventions for which regions enter the likelihood.

    bkg-only : CRs are fitted, SRs and VRs are validation regions.  Without CRs,
               SRs are fitted (as CRs); with only VRs, the VRs are fitted.
    excl/disc: CRs and SRs are fitted, VRs are only evaluated post-fit.
    """
    roles = {e.role for e in entries}
    for e in entries:
        if fit_type == "bkg":
            if "CR" in roles:
                e.fit = e.role == "CR"
            elif "SR" in roles:
                e.fit = e.role == "SR"
            else:
                e.fit = True
        else:
            e.fit = e.role in ("CR", "SR")
    if fit_type in ("excl", "disc"):
        if "SR" not in roles:
            raise ValueError("exclusion/discovery fits need at least one SR")
        if "VR" in roles:
            log.warning("VRs in an exclusion/discovery workspace are not fitted, only shown post-fit")


def _systematics_selection(setting: Any, analysis: Analysis) -> Tuple[List[str], List[str], float]:
    """Returns (groups, individual systematic names, flat uncertainty)."""
    tokens = []
    for tok in as_list(setting):
        tokens += as_list(str(tok))
    groups: List[str] = []
    names: List[str] = []
    flat = 0.0
    known_names = {
        s.name for systs in analysis.systematic_groups.values() for s in systs
    }
    for tok in tokens:
        up = tok.upper()
        if up in ("NONE", "NULL", ""):
            continue
        if up == "ALL":
            groups += analysis.all_groups
        elif up.startswith("DDFLAT"):
            flat = float(tok[6:]) / 100.0
            groups += ["CHFLIP", "FAKES"]
        elif up.startswith("FLAT"):
            flat = float(tok[4:]) / 100.0
        elif up in analysis.systematic_groups or up in ("DDSTAT", "TH"):
            groups.append(up)
        elif tok in known_names:
            names.append(tok)
        else:
            raise ValueError(
                f"unknown systematic group/name '{tok}', groups: {sorted(analysis.systematic_groups)}"
            )
    return unique(groups), unique(names), flat


# ---------------------------------------------------------------------------
# composer
# ---------------------------------------------------------------------------


class Composer:
    def __init__(self, cfg: Dict[str, Any], *, overrides: Optional[Dict[str, Any]] = None):
        cfg = copy.deepcopy(cfg)
        unknown = set(cfg) - {
            "General",
            "Regions",
            "Samples",
            "NormFactors",
            "FixedNF",
            "ExtraSampleWeights",
            "Pruning",
        }
        if unknown:
            raise ValueError(f"unknown top-level keys in workspace YAML: {sorted(unknown)}")
        general = dict(DEFAULT_GENERAL)
        general.update(cfg.get("General", {}))
        general.update(overrides or {})
        unknown_general = set(general) - set(DEFAULT_GENERAL) - {"Name"}
        if unknown_general:
            raise ValueError(f"unknown General keys {sorted(unknown_general)}, allowed: {sorted(DEFAULT_GENERAL)} + Name")
        if "Name" not in general:
            raise ValueError("General.Name is required")
        if general["FitType"] not in FIT_TYPES:
            raise ValueError(f"General.FitType must be one of {FIT_TYPES}")
        self.general = general
        self.cfg = cfg
        self.fit_type = general["FitType"]
        self.analysis = load_analysis(general["Analysis"], general.get("Options") or {})
        self.lumi = float(general["Lumi"])
        self.regions = _parse_regions(cfg.get("Regions", {}))
        _assign_fit_roles(self.regions, self.fit_type)
        self.groups, self.syst_names, self.flat = _systematics_selection(
            general.get("Systematics"), self.analysis
        )
        self._blind(general.get("Blind"))
        self.samples_cfg = cfg.get("Samples", {})
        self.norm_factors = self._parse_normfactors(cfg.get("NormFactors"))
        self.fixed_nf = as_list(cfg.get("FixedNF"))
        self.extra_weights = cfg.get("ExtraSampleWeights") or {}
        self.pruning = cfg.get("Pruning") or {}

    # ------------------------------------------------------------------ helpers
    def _blind(self, blind):
        roles = {r.upper() for r in as_list(blind)}
        if "ALL" in roles:
            roles = set(ROLES)
        bad = roles - set(ROLES)
        if bad:
            raise ValueError(f"General.Blind accepts SR, CR, VR, ALL - got {sorted(bad)}")
        for e in self.regions:
            e.blinded = e.role in roles

    def _parse_normfactors(self, block) -> List[Dict[str, Any]]:
        """``{WZ: CRWZ2j}`` (HistFitter --bkgToNormalize) or a list of dicts."""
        out = []
        if not block:
            return out
        if isinstance(block, dict):
            for sample, regions in block.items():
                out.append({"Sample": sample, "Regions": as_list(regions)})
        else:
            out = [dict(item) for item in block]
        region_names = {r.name for r in self.regions}
        for nf in out:
            nf.setdefault("Name", f"mu_{nf['Sample']}")
            nf.setdefault("Nominal", 1.0)
            nf.setdefault("Bounds", [0.0, 5.0])
            nf["Regions"] = as_list(nf.get("Regions"))
            missing = set(nf["Regions"]) - region_names
            if missing:
                raise ValueError(f"normalisation region(s) {sorted(missing)} of {nf['Sample']} not in the workspace")
        return out

    def _sample_weights(self, sample: Sample) -> List[str]:
        weights = list(sample.weights) if sample.weights is not None else list(self.analysis.nominal_weights)
        weights += list(sample.extra_weights)
        if sample.name in self.extra_weights:
            extra = self.extra_weights[sample.name]
            weights += [str(w) for w in extra] if isinstance(extra, (list, tuple)) else [str(extra)]
        if sample.name in self.fixed_nf:
            nf = self.analysis.normalization_factors.get(sample.name)
            if nf is None:
                raise ValueError(f"FixedNF: no NF for '{sample.name}' in the hard-coded yields JSON")
            weights.append(repr(float(nf[0])))
        return weights

    def _flow(self, entry: RegionEntry, nbins: int) -> Tuple[bool, bool]:
        region = self.analysis.region(entry.name)
        if entry.role in ("CR", "VR"):
            under, over = True, True
        else:
            under, over = False, nbins > 1
        if region.underflow is not None:
            under = region.underflow
        if region.overflow is not None:
            over = region.overflow
        if entry.underflow is not None:
            under = entry.underflow
        if entry.overflow is not None:
            over = entry.overflow
        return under, over

    def _enabled(self, syst) -> bool:
        group = getattr(syst, "group", "")
        if syst.name in self.syst_names:
            return True
        if group in self.groups:
            return True
        # sample-specific groups unknown to the analysis are always on (e.g. MCtemplate)
        return bool(group) and group not in self.analysis.systematic_groups and group not in ("DDSTAT",)

    # ------------------------------------------------------------------ compose
    def compose(self, signal_points: Optional[Sequence[str]] = None) -> Composition:
        analysis = self.analysis
        g = self.general
        input_type = g["InputType"]
        if input_type not in ("ntuple", "histogram"):
            raise ValueError("General.InputType must be 'ntuple' or 'histogram'")

        registry = Registry(
            analysis=analysis,
            input_type=input_type,
            input_path=str(g["InputPath"]),
            lumi=self.lumi,
            missing_tree=g.get("MissingTree", "nominal"),
            lumi_uncertainty=float(g["LumiUncertainty"]),
            stat_constraint=g["StatConstraint"],
        )
        if registry.stat_constraint not in ("Gaussian", "Poisson"):
            raise ValueError("General.StatConstraint must be Gaussian or Poisson")

        # ---------------- regions
        cab_regions = []
        for entry in self.regions:
            region = analysis.region(entry.name)
            variable = entry.variable
            if self.fit_type == "disc" and entry.role == "SR" and variable != CUTS:
                log.warning(f"discovery fits are single-bin: using 'cuts' for {entry.name}")
                variable = entry.variable = CUTS
            binning = region.binning(variable)
            entry.underflow, entry.overflow = self._flow(entry, binning.n)
            registry.regions[entry.name] = entry
            cab_regions.append(
                {
                    "Name": entry.name,
                    "Variable": variable,
                    "Binning": [float(x) for x in binning.bin_edges],
                    "Filter": region.cut,
                }
            )
        region_names = [e.name for e in self.regions]
        registry.blinded_regions = [e.name for e in self.regions if e.blinded]

        # ---------------- samples
        bkg_names = as_list(self.samples_cfg.get("Backgrounds"))
        if not bkg_names:
            raise ValueError("Samples.Backgrounds must list at least one background")
        data_mode = str(self.samples_cfg.get("Data", "ntuple")).lower()
        if data_mode not in ("ntuple", "hardcoded"):
            raise ValueError("Samples.Data must be 'ntuple' or 'hardcoded'")

        samples: List[Sample] = []
        data_sample = analysis.sample(analysis.data_sample)
        samples.append(data_sample)
        for name in bkg_names:
            sample = analysis.sample(name)
            if sample.kind not in (MC, DATADRIVEN):
                raise ValueError(f"'{name}' is not a background sample")
            samples.append(sample)

        # signal
        grid = None
        points: List[SignalPoint] = []
        poi = g["POI"]
        if self.fit_type == "excl":
            sig_cfg = self.samples_cfg.get("Signal")
            if not sig_cfg:
                raise ValueError("exclusion fits need Samples.Signal (Grid, Points)")
            grid = analysis.signal_grid(
                sig_cfg["Grid"], sig_cfg.get("SigNameSchema", "SP:4,LSP:5"), as_list(sig_cfg.get("Exclude"))
            )
            wanted = sig_cfg.get("Points", "all")
            if signal_points:
                wanted = list(signal_points)
            if wanted in ("all", "ALL", None):
                points = list(grid.points)
            else:
                points = [grid.point(str(p)) for p in as_list(wanted)]
            if not points:
                raise ValueError(f"no signal points selected from grid {grid.name}")
            for point in points:
                samples.append(analysis.signal_sample(point, grid))
            registry.signal_samples = [p.name for p in points]
        elif self.fit_type == "disc":
            srs = [e.name for e in self.regions if e.role == "SR"]
            if len(srs) != 1:
                raise ValueError("discovery fits need exactly one SR")
            disc = Sample(
                name=f"DiscoveryMode_{srs[0]}",
                kind=DISCOVERY,
                color="#e7298a",
                label="Discovery signal",
                staterror=False,
                norm_by_theory=False,
                experimental_systematics="none",
            )
            samples.append(disc)
            registry.custom[(disc.name, "", None)] = CustomTemplate(
                kind="yields", yields=lambda region, nbins: (np.ones(nbins), np.zeros(nbins))
            )
            registry.signal_samples = []

        free_nf_samples = {nf["Sample"]: nf for nf in self.norm_factors}
        mc_backgrounds = [n for n in bkg_names if analysis.sample(n).kind == MC]
        for name in self.fixed_nf:
            if name not in mc_backgrounds:
                raise ValueError(f"FixedNF: '{name}' is not an MC background of this workspace ({mc_backgrounds})")
            if name not in analysis.normalization_factors:
                raise ValueError(f"FixedNF: no NF for '{name}' in the hard-coded yields JSON")
        for name in self.extra_weights:
            if name not in mc_backgrounds:
                raise ValueError(f"ExtraSampleWeights: '{name}' is not an MC background of this workspace")
        for nf in self.norm_factors:
            if nf["Sample"] not in bkg_names:
                raise ValueError(f"NormFactors: sample {nf['Sample']} is not among the backgrounds")

        cab_samples = []
        for sample in samples:
            registry.samples[sample.name] = sample
            entry: Dict[str, Any] = {"Name": sample.name}
            if sample.kind == DATA:
                entry["Data"] = True
                if data_mode == "hardcoded":
                    registry.custom[(sample.name, "", None)] = CustomTemplate(
                        kind="yields", yields=self._hardcoded_data()
                    )
                else:
                    entry["Tree"] = sample.tree_name()
                    entry["SamplePath"] = sample.file_name(self.lumi)
            elif sample.kind in (MC, SIGNAL):
                weights = self._sample_weights(sample)
                registry.sample_weights[sample.name] = weights
                entry["Tree"] = sample.tree_name()
                entry["SamplePath"] = sample.file_name(self.lumi)
                entry["Weight"] = expressions.product(weights)
            elif sample.kind == DATADRIVEN:
                entry["DisableStaterror"] = True
                registry.custom[(sample.name, "", None)] = CustomTemplate(
                    kind="yields", yields=self._dd_yields(sample, "nominal")
                )
            elif sample.kind == DISCOVERY:
                entry["Regions"] = [e.name for e in self.regions if e.role == "SR"]
            if not sample.staterror and "DisableStaterror" not in entry and sample.kind != DATA:
                entry["DisableStaterror"] = True
            cab_samples.append(entry)
            # luminosity uncertainty (HistFitter setNormByTheory)
            if sample.kind in (MC, SIGNAL) and sample.norm_by_theory and sample.name not in free_nf_samples:
                registry.lumi_samples.append(sample.name)

        # ---------------- normalisation factors
        bounds = g.get("POIBounds") or ([0.0, 100.0] if self.fit_type == "disc" else [0.0, 10.0])
        cab_nfs = []
        if self.fit_type == "excl":
            cab_nfs.append(
                {"Name": poi, "Samples": registry.signal_samples, "Nominal": 1.0, "Bounds": [float(b) for b in bounds]}
            )
        elif self.fit_type == "disc":
            cab_nfs.append(
                {
                    "Name": poi,
                    "Samples": [s.name for s in samples if s.kind == DISCOVERY],
                    "Nominal": 1.0,
                    "Bounds": [float(b) for b in bounds],
                }
            )
        else:
            # HistFitter always declares the POI; it acts on no sample in bkg-only fits
            cab_nfs.append({"Name": poi, "Samples": ["signal"], "Nominal": 1.0, "Bounds": [float(b) for b in bounds]})
        for nf in self.norm_factors:
            cab_nfs.append(
                {
                    "Name": nf["Name"],
                    "Samples": [nf["Sample"]],
                    "Nominal": float(nf["Nominal"]),
                    "Bounds": [float(b) for b in nf["Bounds"]],
                }
            )

        # ---------------- systematics
        cab_systs: Dict[str, Dict[str, Any]] = {}
        groups_of: Dict[str, str] = {}

        def add_syst(entry: Dict[str, Any], sample: Sample, group: str):
            name = entry["Name"]
            if name in sample.exclude_systematics:
                return
            if name in cab_systs:
                existing = cab_systs[name]
                comparable = {k: v for k, v in entry.items() if k != "Samples"}
                if {k: v for k, v in existing.items() if k != "Samples"} != comparable:
                    raise ValueError(f"systematic '{name}' defined differently for different samples")
                if sample.name not in existing["Samples"]:
                    existing["Samples"].append(sample.name)
            else:
                entry = dict(entry)
                entry["Samples"] = [sample.name]
                cab_systs[name] = entry
            groups_of[entry.get("ModifierName", name)] = group or "OTHER"
            nf = free_nf_samples.get(sample.name)
            if (
                nf is not None
                and g.get("NormalizeSystematicsInCR", True)
                and entry["Type"] == "NormPlusShape"
            ):
                registry.normalize_to.setdefault(name, {})[sample.name] = list(nf["Regions"])

        sample_regions = {
            s["Name"]: as_list(s.get("Regions")) or region_names for s in cab_samples
        }

        for sample in samples:
            if sample.kind in (DATA, DISCOVERY):
                continue
            is_signal = sample.kind == SIGNAL
            if sample.kind in (MC, SIGNAL):
                systs: List[Any] = []
                if sample.experimental_systematics == "all":
                    systs += [
                        s
                        for s in self.analysis.systematics_for_groups(list(self.analysis.systematic_groups))
                        if self._enabled(s)
                    ]
                systs += [s for s in sample.extra_systematics if self._enabled(s)]
                if not is_signal and "TH" in self.groups:
                    systs += list(self.analysis.theory.get(sample.name, []))
                for syst in systs:
                    if isinstance(syst, WeightSyst):
                        if syst.applies_to == "bkg" and is_signal or syst.applies_to == "signal" and not is_signal:
                            continue
                        if syst.replace is not None and syst.replace not in registry.sample_weights[sample.name]:
                            log.debug(f"{syst.name}: '{syst.replace}' not in weights of {sample.name}, skipped")
                            continue
                        add_syst(self._weight_syst(syst), sample, syst.group)
                    elif isinstance(syst, TreeSyst):
                        add_syst(self._tree_syst(syst), sample, syst.group)
                    elif isinstance(syst, TheorySyst):
                        entry = self._theory_syst(
                            syst, sample, registry, sample_regions[sample.name], sample.name in free_nf_samples
                        )
                        if entry is not None:
                            add_syst(entry, sample, syst.group)
                    elif isinstance(syst, NormSyst):
                        add_syst(
                            {"Name": syst.name, "Type": "Normalization", "Up": {"Normalization": syst.up},
                             "Down": {"Normalization": syst.down}},
                            sample,
                            syst.group,
                        )
                    else:
                        raise TypeError(f"unknown systematic type {type(syst)}")
                # fixed NF uncertainty (HistFitter --NF)
                if sample.name in self.fixed_nf:
                    nf = self.analysis.normalization_factors[sample.name]
                    add_syst(
                        {"Name": f"NF_{sample.name}", "Type": "Normalization",
                         "Up": {"Normalization": float(nf[1])}, "Down": {"Normalization": -float(nf[2])}},
                        sample,
                        "NF",
                    )
                # flat uncertainty on the backgrounds (FLAT<n>)
                if self.flat and not is_signal:
                    add_syst(
                        {"Name": f"FlatSyst_{sample.name}", "Type": "Normalization",
                         "Up": {"Normalization": self.flat}, "Down": {"Normalization": -self.flat}},
                        sample,
                        "FLAT",
                    )
            elif sample.kind == DATADRIVEN:
                self._dd_systs(sample, registry, add_syst, region_names)

        # ---------------- assemble
        out_dir = pathlib.Path(g["OutputDir"]) / g["Name"]
        if input_type == "ntuple":
            input_path = str(pathlib.Path(g["InputPath"]) / "{SamplePath}")
        else:
            input_path = str(g["InputPath"])
        config = {
            "General": {
                "Measurement": g["Measurement"],
                "POI": poi if self.fit_type in ("excl", "disc") else "",
                "InputPath": input_path,
                "HistogramFolder": str(out_dir / "histograms"),
            },
            "Regions": cab_regions,
            "Samples": cab_samples,
            "NormFactors": cab_nfs,
            "Systematics": list(cab_systs.values()),
        }
        meta = self._meta(samples, points, grid, groups_of, registry, cab_nfs)
        return Composition(config=config, registry=registry, meta=meta, signal_points=points, grid=grid)

    # ------------------------------------------------------------------ syst builders
    @staticmethod
    def _tree_syst(syst: TreeSyst) -> Dict[str, Any]:
        entry = {"Name": syst.name, "Type": "NormPlusShape", "Up": {"Tree": f"{{prefix}}_{syst.up}"}}
        if syst.down is None:
            entry["Down"] = {"Symmetrize": True}
        else:
            entry["Down"] = {"Tree": f"{{prefix}}_{syst.down}"}
        return entry

    @staticmethod
    def _weight_syst(syst: WeightSyst) -> Dict[str, Any]:
        def expr(var):
            if syst.replace is not None:
                return f"{{weight:{syst.replace}->{var}}}"
            return f"{{weight}}*{expressions._wrap(var)}"

        entry = {"Name": syst.name, "Type": "NormPlusShape", "Up": {"Weight": expr(syst.up)}}
        entry["Down"] = {"Symmetrize": True} if syst.down is None else {"Weight": expr(syst.down)}
        return entry

    def _theory_values(self, key: str, direction: str) -> Optional[Dict[str, Any]]:
        return self.analysis.theory_values.get(f"thUnc_{key}{direction}")

    def _theory_syst(
        self, syst: TheorySyst, sample: Sample, registry: Registry, regions: List[str], normalized: bool = False
    ):
        up_vals = self._theory_values(syst.key, "Up")
        down_vals = self._theory_values(syst.key, "Down")
        if up_vals is None or down_vals is None:
            log.debug(f"theory uncertainty thUnc_{syst.key}Up/Down not found, {syst.name} skipped")
            return None
        binned = bool(self.general.get("BinnedTheoryUncertainty")) and syst.allow_binned
        if syst.binned_only_if_normalized and not normalized:
            binned = False

        def factor(values, sign):
            def _f(region: str, nbins: int) -> np.ndarray:
                entry = registry.regions[region]
                reg_vals = values.get(region)
                if reg_vals is None:
                    log.warning(f"{syst.name}: no value for region {region}, no effect assigned")
                    return np.ones(nbins)
                var_vals = reg_vals.get(entry.variable, reg_vals.get(entry.variable.replace("/", "")))
                if binned and var_vals is not None and len(var_vals) == nbins:
                    rel = np.asarray(var_vals, dtype=float)
                else:
                    if binned and nbins > 1:
                        log.warning(f"{syst.name}: no binned values for {region}/{entry.variable}, using 'cuts'")
                    if CUTS not in reg_vals:
                        log.warning(f"{syst.name}: no 'cuts' value for region {region}, no effect assigned")
                        return np.ones(nbins)
                    rel = np.full(nbins, float(reg_vals[CUTS][0]))
                return 1.0 + sign * rel

            return _f

        f_up, f_down = factor(up_vals, +1.0), factor(down_vals, -1.0)
        if syst.selection:
            # only events passing the selection are varied (weight-level, single value per region)
            sel = syst.selection

            def weight(f):
                return lambda region: (
                    f"{{weight}}*where({expressions.translate(sel)}, {float(f(region, 1)[0])!r}, 1.0)"
                )

            registry.custom[(sample.name, syst.name, "Up")] = CustomTemplate(kind="ntuple", weight=weight(f_up))
            registry.custom[(sample.name, syst.name, "Down")] = CustomTemplate(kind="ntuple", weight=weight(f_down))
        else:
            registry.custom[(sample.name, syst.name, "Up")] = CustomTemplate(kind="scale", factors=f_up)
            registry.custom[(sample.name, syst.name, "Down")] = CustomTemplate(kind="scale", factors=f_down)
        return {"Name": syst.name, "Type": "NormPlusShape", "Up": {}, "Down": {}}

    # ------------------------------------------------------------------ data-driven
    def _yield_table(self, key: str) -> Dict[str, Any]:
        table = self.analysis.hardcoded_yields.get(key)
        if table is None:
            raise KeyError(f"'{key}' not found in the hard-coded yields JSON")
        return table

    def _lookup(self, table: Dict[str, Any], region: str, nbins: int, what: str) -> Optional[np.ndarray]:
        variable = self.general_variable(region)
        entry = table.get(region, {})
        values = entry.get(variable)
        if values is None:
            # HistFitter JSON convention: "met/ht_lept" is stored as "metht_lept"
            values = entry.get(variable.replace("/", ""))
        if values is None:
            return None
        values = np.asarray(values, dtype=float)
        if len(values) != nbins:
            raise ValueError(f"{what}: {len(values)} values for {region}/{variable} but {nbins} bins")
        return values

    def general_variable(self, region: str) -> str:
        for e in self.regions:
            if e.name == region:
                return e.variable
        raise KeyError(region)

    def _hardcoded_data(self):
        table = self._yield_table("dataCounts")

        def _f(region, nbins):
            values = self._lookup(table, region, nbins, "dataCounts")
            if values is None:
                raise ValueError(f"no hard-coded data for region {region}/{self.general_variable(region)}")
            return values, values.copy()

        return _f

    def _dd_yields(self, sample: Sample, which: str):
        spec = sample.yields
        nominal = self._yield_table(spec.nominal)
        stat = self._yield_table(spec.stat) if spec.stat else {}
        up = self._yield_table(spec.syst_up) if spec.syst_up else {}
        down = self._yield_table(spec.syst_down) if spec.syst_down else {}

        def _get(region, nbins):
            y = self._lookup(nominal, region, nbins, f"{sample.name} yields")
            if y is None:
                msg = f"no {sample.name} yields ('{spec.nominal}') for {region}/{self.general_variable(region)}"
                if self.general.get("MissingYields", "error") != "zero":
                    raise ValueError(msg + " in the hard-coded yields JSON (General.MissingYields: zero to use 0)")
                log.warning(msg + ": set to 0")
                y = np.zeros(nbins)
            s = self._lookup(stat, region, nbins, f"{sample.name} stat") if stat else None
            s = np.zeros(nbins) if s is None else np.abs(s)
            return y, s

        if which == "nominal":
            return lambda region, nbins: (_get(region, nbins)[0], _get(region, nbins)[1] ** 2)
        if which in ("stat_up", "stat_down"):
            sign = 1.0 if which == "stat_up" else -1.0

            def _stat(region, nbins):
                y, s = _get(region, nbins)
                # HistFitter: relative variation 1 +- stat/yield, no effect where yield = 0.
                # If the uncertainty is (almost) as large as the yield, y - s would be
                # <= 0; the log-normal-like y^2/(y + s) is used instead (y - s is
                # ill-defined there, HistFactory clamps the total at zero).
                varied = y + sign * s
                if sign < 0:
                    safe = np.divide(y * y, y + s, out=np.zeros_like(y), where=(y + s) > 0)
                    varied = np.where(varied < 0.1 * y, safe, varied)
                return np.where(y > 0, varied, y), np.zeros(nbins)

            return _stat
        if which in ("syst_up", "syst_down"):
            table = up if which == "syst_up" else down
            sign = 1.0 if which == "syst_up" else -1.0

            def _f(region, nbins):
                y = _get(region, nbins)[0]
                rel = self._lookup(table, region, nbins, f"{sample.name} {which}")
                rel = np.zeros(nbins) if rel is None else rel
                return y * (1.0 + sign * rel), np.zeros(nbins)

            return _f
        raise ValueError(which)

    def _dd_systs(self, sample: Sample, registry: Registry, add_syst, region_names: List[str]):
        spec = sample.yields
        if spec is None:
            return
        syst_on = spec.group in self.groups if spec.group else False
        stat_on = syst_on or "DDSTAT" in self.groups
        for region in region_names:
            if syst_on and spec.syst_up and spec.syst_down:
                name = f"syst_{sample.name}_{region}"
                registry.custom[(sample.name, name, "Up")] = CustomTemplate(kind="yields", yields=self._dd_yields(sample, "syst_up"))
                registry.custom[(sample.name, name, "Down")] = CustomTemplate(kind="yields", yields=self._dd_yields(sample, "syst_down"))
                add_syst({"Name": name, "Type": "NormPlusShape", "Regions": [region], "Up": {}, "Down": {}}, sample, spec.group)
            if stat_on and spec.stat:
                name = f"stat_{sample.name}_{region}"
                registry.custom[(sample.name, name, "Up")] = CustomTemplate(kind="yields", yields=self._dd_yields(sample, "stat_up"))
                registry.custom[(sample.name, name, "Down")] = CustomTemplate(kind="yields", yields=self._dd_yields(sample, "stat_down"))
                add_syst({"Name": name, "Type": "NormPlusShape", "Regions": [region], "Up": {}, "Down": {}}, sample, "DDSTAT")
                if registry.stat_constraint == "Poisson":
                    nbins = registry.analysis.region(region).binning(self.general_variable(region)).n
                    _, stat = self._dd_yields(sample, "nominal")(region, nbins)
                    registry.poisson_stat.setdefault(name, {})[sample.name] = list(np.sqrt(stat))

    # ------------------------------------------------------------------ metadata
    def _meta(self, samples, points, grid, groups_of, registry, cab_nfs) -> Dict[str, Any]:
        import cabinetry
        import pyhf

        from fit2l3l import __version__

        g = self.general
        regions = []
        for e in self.regions:
            region = self.analysis.region(e.name)
            binning = region.binning(e.variable)
            regions.append(
                {
                    "name": e.name,
                    "role": e.role,
                    "fit": e.fit,
                    "blinded": e.blinded,
                    "variable": e.variable,
                    "label": region.label(e.variable),
                    "title": region.title,
                    "bins": [float(x) for x in binning.bin_edges],
                    "overflow": e.overflow,
                    "underflow": e.underflow,
                }
            )
        meta = {
            "name": g["Name"],
            "analysis": g["Analysis"],
            "fit_type": self.fit_type,
            "lumi_pb": self.lumi,
            "lumi_fb": self.lumi / 1000.0,
            "lumi_uncertainty": float(g["LumiUncertainty"]),
            "poi": g["POI"],
            "measurement": g["Measurement"],
            "systematics": {"groups": self.groups, "names": self.syst_names, "flat": self.flat},
            "regions": regions,
            "samples": {
                s.name: {"kind": s.kind, "color": s.color, "label": s.label}
                for s in samples
                if s.kind != SIGNAL
            },
            "normfactors": [nf["Name"] for nf in cab_nfs],
            "systematic_groups": groups_of,
            "blinded_regions": registry.blinded_regions,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "versions": {"fit2l3l": __version__, "cabinetry": cabinetry.__version__, "pyhf": pyhf.__version__},
        }
        if grid is not None:
            meta["signal"] = {
                "grid": grid.name,
                "mass_labels": list(grid.mass_labels),
                "points": [
                    {
                        "name": p.name,
                        "dsid": p.dsid,
                        "masses": list(p.masses),
                        "xsec": p.xsec,
                        "xsec_rel_up": p.xsec_rel_up,
                        "xsec_rel_down": p.xsec_rel_down,
                    }
                    for p in points
                ],
            }
            meta["signal_style"] = {"color": "#e7298a"}
        return meta


def compose(cfg: Dict[str, Any], *, overrides: Optional[Dict[str, Any]] = None, signal_points=None) -> Composition:
    return Composer(cfg, overrides=overrides).compose(signal_points=signal_points)
