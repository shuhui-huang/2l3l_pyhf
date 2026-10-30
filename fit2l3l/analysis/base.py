"""Framework-level building blocks used to describe an analysis in python.

An *analysis module* (e.g. :mod:`fit2l3l.analysis.ss3l`) uses these classes to
define the physics content:

* :class:`Region`       - selection (cut string), binning options, labels
* :class:`Sample`       - ntuple/tree naming, weights, colours, data-driven yields
* :class:`TreeSyst`     - tree-based (kinematic) systematic, one tree per variation
* :class:`WeightSyst`   - weight-based systematic (replace / append weight factors)
* :class:`TheorySyst`   - user-defined relative uncertainty per region (and bin),
  optionally restricted to a subset of events (e.g. some DSIDs)
* :class:`SignalPoint` / :class:`SignalGrid` - signal models

The YAML configuration only *selects* regions/samples/systematic groups; the
definitions themselves live in python (like ``fitConfigSS3L.py`` did for
HistFitter).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------

#: name of the special "variable" meaning "single bin, count all events"
CUTS = "cuts"


@dataclass
class Binning:
    """Binning of a variable in a region.

    Either equal-width ``(nbins, low, high)`` (HistFitter convention) or explicit
    ``edges``.
    """

    nbins: Optional[int] = None
    low: Optional[float] = None
    high: Optional[float] = None
    edges: Optional[Sequence[float]] = None

    @classmethod
    def from_any(cls, spec) -> "Binning":
        if isinstance(spec, Binning):
            return spec
        if isinstance(spec, dict):
            return cls(**spec)
        spec = list(spec)
        if len(spec) == 3 and float(spec[0]).is_integer() and spec[0] >= 1 and spec[2] > spec[1]:
            return cls(nbins=int(spec[0]), low=float(spec[1]), high=float(spec[2]))
        return cls(edges=[float(x) for x in spec])

    @property
    def bin_edges(self) -> np.ndarray:
        if self.edges is not None:
            return np.asarray(self.edges, dtype=float)
        return np.linspace(self.low, self.high, self.nbins + 1)

    @property
    def n(self) -> int:
        return len(self.bin_edges) - 1


@dataclass
class Region:
    """A selection region.

    Args:
        name: region name (used as channel name in the workspace)
        cut: selection, C++ (``&&``, ``||``, ``!``) or python syntax
        binnings: variable expression -> binning.  ``"cuts"`` (single bin) is
            always available.
        title: label used in plots/tables
        variable_labels: optional pretty labels for the variables
        overflow / underflow: include over/underflow in the last/first bin.
            ``None`` means "use the default for the role the region plays in the
            fit" (HistFitter: CR/VR always, SR only if binned).
    """

    name: str
    cut: str
    binnings: Dict[str, Binning] = field(default_factory=dict)
    title: str = ""
    variable_labels: Dict[str, str] = field(default_factory=dict)
    overflow: Optional[bool] = None
    underflow: Optional[bool] = None

    def __post_init__(self):
        self.binnings = {k: Binning.from_any(v) for k, v in self.binnings.items()}
        self.binnings.setdefault(CUTS, Binning(nbins=1, low=0.5, high=1.5))
        if not self.title:
            self.title = self.name

    def binning(self, variable: str = CUTS) -> Binning:
        if variable not in self.binnings:
            raise KeyError(
                f"variable '{variable}' has no binning defined for region '{self.name}', "
                f"available: {sorted(self.binnings)}"
            )
        return self.binnings[variable]

    def label(self, variable: str) -> str:
        if variable == CUTS:
            return self.title
        return self.variable_labels.get(variable, variable)


# ---------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------

#: sample kinds
MC, DATADRIVEN, DATA, SIGNAL, DISCOVERY = "mc", "datadriven", "data", "signal", "discovery"


@dataclass
class DataDrivenYields:
    """Keys into the hard-coded yields JSON for a data-driven background.

    ``nominal`` and ``stat`` are absolute, ``syst_up``/``syst_down`` relative (as in
    ``hardCodedYields.json`` of the HistFitter framework).
    """

    nominal: str
    stat: Optional[str] = None
    syst_up: Optional[str] = None
    syst_down: Optional[str] = None
    #: systematics group switching on the systematic (e.g. "FAKES")
    group: Optional[str] = None


@dataclass
class Sample:
    """A sample (background, data or signal).

    ``file`` and ``tree`` may contain ``{name}`` and ``{lumi}`` placeholders.
    ``weights`` is a list of factors multiplied together; ``None`` means the
    analysis default nominal weights.
    """

    name: str
    kind: str = MC
    color: str = "#999999"
    label: str = ""
    file: str = "{name}.{lumi}.root"
    tree: str = "{name}_nom"
    tree_prefix: Optional[str] = None  # prefix of systematic trees, default: name
    weights: Optional[List[str]] = None
    extra_weights: List[str] = field(default_factory=list)
    yields: Optional[DataDrivenYields] = None
    #: include in the MC statistical uncertainty (staterror / gamma parameters)
    staterror: bool = True
    #: luminosity uncertainty applies (HistFitter setNormByTheory)
    norm_by_theory: bool = True
    #: which experimental systematics apply: "all" or "none" (only sample-specific)
    experimental_systematics: str = "all"
    #: names of systematics (cabinetry names / modifiers) not applied to this sample
    exclude_systematics: List[str] = field(default_factory=list)
    #: sample specific additional systematics (WeightSyst/TreeSyst/TheorySyst objects)
    extra_systematics: List[object] = field(default_factory=list)
    #: free metadata (e.g. masses of a signal point)
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.label:
            self.label = self.name

    @property
    def is_mc(self) -> bool:
        return self.kind in (MC, SIGNAL)

    def file_name(self, lumi: Union[str, float]) -> str:
        return self.file.format(name=self.name, lumi=_lumi_tag(lumi))

    def tree_name(self) -> str:
        return self.tree.format(name=self.name)

    def syst_tree_prefix(self) -> str:
        return self.tree_prefix if self.tree_prefix is not None else self.name

    def replace(self, **kwargs) -> "Sample":
        return dataclasses.replace(self, **kwargs)


def _lumi_tag(lumi) -> str:
    try:
        value = float(lumi)
        return str(int(value)) if value.is_integer() else str(value)
    except (TypeError, ValueError):
        return str(lumi)


# ---------------------------------------------------------------------------
# systematics
# ---------------------------------------------------------------------------


@dataclass
class TreeSyst:
    """Kinematic systematic: the variation is read from a different tree.

    The tree name is ``<tree_prefix>_<up>`` (HistFitter: ``<sample>_<suffix>``).
    ``down=None`` means one-sided: the down variation is obtained by
    symmetrisation (HistFitter ``histoSysOneSideSym``).
    """

    name: str
    up: str
    down: Optional[str] = None
    group: str = ""


@dataclass
class WeightSyst:
    """Weight systematic.

    Either replace a factor of the nominal weight (``replace="wmu_nom"`` with
    ``up="wmu_stat_up"``), or multiply the nominal weight by an extra factor
    (``replace=None``, ``up="wpdf_up"``).  ``down=None`` -> symmetrised.
    """

    name: str
    up: str
    down: Optional[str] = None
    replace: Optional[str] = None
    group: str = ""
    #: "all", "bkg" or "signal"
    applies_to: str = "all"


@dataclass
class TheorySyst:
    """Relative uncertainty taken from a JSON table (``thUnc_<key>Up/Down``).

    The JSON layout is ``{"thUnc_<key>Up": {region: {variable: [v_bin...]}}}`` where
    ``"cuts"`` holds the single-bin value.  The up (down) template is
    ``nominal * (1 + up)`` (``nominal * (1 - down)``).

    ``selection``: restrict the variation to a subset of events (e.g. some DSIDs);
    other events are unchanged.  Requires reading the ntuple again.
    """

    name: str
    key: str
    selection: Optional[str] = None
    allow_binned: bool = True
    group: str = "TH"


@dataclass
class NormSyst:
    """Fixed normalisation uncertainty (e.g. a flat 30% or an NF uncertainty)."""

    name: str
    up: float
    down: float
    group: str = ""


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------


@dataclass
class SignalPoint:
    dsid: str
    masses: Tuple[float, ...]
    name: str = ""
    physics_short: str = ""
    xsec: Optional[float] = None  # [fb], cross-section * k-factor (* filter eff.)
    xsec_rel_up: Optional[float] = None
    xsec_rel_down: Optional[float] = None
    extra_weights: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            self.name = f"signal{self.dsid}"

    @property
    def m0(self) -> float:
        return self.masses[0]

    @property
    def m12(self) -> float:
        return self.masses[1] if len(self.masses) > 1 else 0.0


@dataclass
class SignalGrid:
    name: str
    points: List[SignalPoint]
    mass_labels: Tuple[str, ...] = ("m0", "m12")

    def point(self, key: str) -> SignalPoint:
        for p in self.points:
            if key in (p.dsid, p.name):
                return p
        raise KeyError(f"signal point '{key}' not found in grid {self.name}")


# ---------------------------------------------------------------------------
# the analysis container
# ---------------------------------------------------------------------------


@dataclass
class Analysis:
    """Everything the workspace builder needs to know about the physics.

    Analysis modules return an instance of this class from ``build_analysis``.
    """

    name: str
    regions: Dict[str, Region]
    samples: Dict[str, Sample]
    nominal_weights: List[str]
    #: experimental systematics per group, e.g. {"JET": [TreeSyst...], "MU": [...]}
    systematic_groups: Dict[str, List[object]]
    #: groups switched on by "ALL"
    all_groups: List[str]
    #: theory systematics per sample name (group "TH")
    theory: Dict[str, List[object]] = field(default_factory=dict)
    #: relative theory uncertainties ("thUnc_<key>Up/Down" -> region -> var -> values)
    theory_values: Dict[str, dict] = field(default_factory=dict)
    #: hard-coded yields (data-driven backgrounds, data counts), JSON content
    hardcoded_yields: Dict[str, dict] = field(default_factory=dict)
    #: fixed normalisation factors {sample: [nominal, rel_up, rel_down]}
    normalization_factors: Dict[str, List[float]] = field(default_factory=dict)
    #: callable(grid_name, sig_name_schema, options) -> SignalGrid
    signal_grid: Optional[Callable[..., SignalGrid]] = None
    #: callable(point, grid) -> Sample  (signal sample definition)
    signal_sample: Optional[Callable[..., Sample]] = None
    #: callable(region, sample, systematic_name, template) -> "path/in/file" for
    #: histogram inputs (InputType: histogram)
    histogram_path: Optional[Callable[..., str]] = None
    #: name of the data sample used when Data: ntuple
    data_sample: str = "data"
    #: plotting configuration of signal models (contours)
    plotting_models: Dict[str, dict] = field(default_factory=dict)
    #: systematic name -> group (for tables), filled automatically
    options: Dict[str, object] = field(default_factory=dict)

    def region(self, name: str) -> Region:
        try:
            return self.regions[name]
        except KeyError:
            raise KeyError(
                f"region '{name}' is not defined in analysis '{self.name}' "
                f"(define it in the analysis python module)"
            ) from None

    def sample(self, name: str) -> Sample:
        try:
            return self.samples[name]
        except KeyError:
            raise KeyError(
                f"sample '{name}' is not defined in analysis '{self.name}' "
                f"(define it in the analysis python module)"
            ) from None

    def systematics_for_groups(self, groups: Sequence[str]) -> List[object]:
        out = []
        for group in groups:
            for syst in self.systematic_groups.get(group, []):
                if not getattr(syst, "group", ""):
                    syst.group = group
                out.append(syst)
        return out
