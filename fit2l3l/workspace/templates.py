"""Template histogram production for cabinetry.

cabinetry's ``templates.build`` is driven with a :class:`cabinetry.route.Router`
whose template builder returns histograms computed here.  Compared to the
default uproot backend of cabinetry this

* reads every (file, tree) only once and fills all histograms needed from it
  (all regions, all weight variations) - important with O(100) systematics
* understands the C++ selection strings of the HistFitter configuration
  (:mod:`fit2l3l.workspace.expressions`)
* adds under-/overflow to the first/last bin (HistFitter ``useOverflowBin``)
* builds templates that cannot be expressed in a cabinetry config: data-driven
  backgrounds and hard-coded data from JSON yields, the discovery dummy signal,
  region/bin dependent relative (theory) uncertainties, ...
* supports histogram inputs (``InputType: histogram``)
"""

from __future__ import annotations

import collections
import concurrent.futures
import copy
import glob
import logging
import multiprocessing
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import boost_histogram as bh
import cabinetry
import numpy as np

from fit2l3l.analysis.base import CUTS
from fit2l3l.workspace import expressions
from fit2l3l.workspace.composer import Composition, CustomTemplate

log = logging.getLogger(__name__)

_WEIGHT_PLACEHOLDER = re.compile(r"\{weight(?::([^{}]+?)->([^{}]+?))?\}")
#: yields below this value are set to it (pyhf cannot handle negative expectations)
MIN_YIELD = 1e-6


@dataclass
class NtupleJob:
    name: str  # cabinetry histogram name
    sample: str
    systematic: str
    template: Optional[str]
    edges: Tuple[float, ...]
    variable: str
    selection: str
    weight: str
    underflow: bool
    overflow: bool
    nominal_name: str = ""
    is_data: bool = False


@dataclass
class TreeGroup:
    paths: Tuple[str, ...]
    tree: str
    jobs: List[NtupleJob] = field(default_factory=list)
    nominal: bool = False  # contains nominal jobs (tree must exist)


def _fill(x, w, edges, underflow, overflow, single_bin):
    nbins = len(edges) - 1
    if single_bin:
        idx = np.zeros(len(w), dtype=int)
    else:
        x = np.asarray(x, dtype=float)
        good = ~np.isnan(x)
        x, w = x[good], w[good]
        idx = np.searchsorted(edges, x, side="right") - 1
        # a value exactly on the last edge belongs to the last bin (like np.histogram)
        idx[(x == edges[-1])] = nbins - 1
        if underflow:
            idx[idx < 0] = 0
        if overflow:
            idx[idx >= nbins] = nbins - 1
        keep = (idx >= 0) & (idx < nbins)
        idx, w = idx[keep], w[keep]
    sumw = np.bincount(idx, weights=w, minlength=nbins).astype(float)
    sumw2 = np.bincount(idx, weights=w * w, minlength=nbins).astype(float)
    return sumw, sumw2


def _read_tree(paths: Sequence[str], tree: str, branches: Sequence[str]):
    """Returns (arrays, n_entries) or None if the tree is missing in all files."""
    import uproot

    arrays: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    found = False
    n_total = 0
    for pattern in paths:
        files = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        if not files:
            raise FileNotFoundError(f"no input file matches {pattern}")
        for path in files:
            if not pathlib.Path(path).exists() and "://" not in path:
                raise FileNotFoundError(f"input file {path} does not exist")
            with uproot.open(path) as handle:
                if tree not in handle:
                    continue
                found = True
                obj = handle[tree]
                available = set(obj.keys())
                missing = [b for b in branches if b not in available]
                if missing:
                    raise KeyError(f"branches {missing} not found in {path}:{tree}")
                n_total += obj.num_entries
                if branches:
                    data = obj.arrays(list(branches), library="np")
                    for b in branches:
                        arrays[b].append(np.asarray(data[b]))
    if not found:
        return None
    out = {b: (np.concatenate(v) if v else np.zeros(0)) for b, v in arrays.items()}
    return out, n_total


def process_group(group: TreeGroup) -> Dict[str, Any]:
    """Fills all histograms of a (files, tree) group.  Returns name -> (sumw, sumw2)
    or ``{"__missing__": True}`` if the tree does not exist."""
    exprs = set()
    for job in group.jobs:
        exprs.add(job.weight)
        if job.selection:
            exprs.add(job.selection)
        if job.variable != CUTS:
            exprs.add(job.variable)
    branches = sorted(expressions.symbols_of(exprs))
    result = _read_tree(group.paths, group.tree, branches)
    if result is None:
        return {"__missing__": True}
    arrays, n_entries = result
    cache: Dict[str, np.ndarray] = {}

    def ev(expr: str) -> np.ndarray:
        if expr not in cache:
            cache[expr] = expressions.evaluate(expr, arrays, n_entries)
        return cache[expr]

    out = {}
    for job in group.jobs:
        if n_entries == 0:
            nb = len(job.edges) - 1
            out[job.name] = (np.zeros(nb), np.zeros(nb))
            continue
        mask = ev(job.selection).astype(bool) if job.selection else np.ones(n_entries, dtype=bool)
        w = np.asarray(ev(job.weight), dtype=float)[mask]
        x = None if job.variable == CUTS else ev(job.variable)[mask]
        out[job.name] = _fill(
            x, w, np.asarray(job.edges), job.underflow, job.overflow, job.variable == CUTS
        )
    return out


class TemplateEngine:
    """Plans, computes and hands histograms to cabinetry."""

    def __init__(self, composition: Composition, *, n_workers: int = 1):
        self.comp = composition
        self.reg = composition.registry
        self.n_workers = max(1, int(n_workers))
        self.groups: Dict[Tuple[Tuple[str, ...], str], TreeGroup] = {}
        self.post: List[Tuple[str, Dict, Dict, Dict, Optional[str], CustomTemplate]] = []
        self.hist_jobs: List[Tuple[str, Dict, Dict, Dict, Optional[str]]] = []
        self.store: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.fallback_to_nominal: Dict[str, str] = {}
        self.sample_of: Dict[str, str] = {}
        self._planned = False
        self._warned = set()

    # ------------------------------------------------------------------ resolution
    def _weight(self, sample: str, expr: Optional[str]) -> str:
        nominal = self.reg.sample_weights.get(sample, [])
        if expr is None:
            return expressions.product(nominal)

        def repl(match: re.Match) -> str:
            old, new = match.group(1), match.group(2)
            if old is None:
                return expressions._wrap(expressions.product(nominal))
            old, new = old.strip(), new.strip()
            if old not in nominal:
                key = (sample, old)
                if key not in self._warned:
                    log.warning(f"weight factor '{old}' not in nominal weight of {sample}: no variation")
                    self._warned.add(key)
                return expressions._wrap(expressions.product(nominal))
            return expressions._wrap(expressions.product([new if f == old else f for f in nominal]))

        return _WEIGHT_PLACEHOLDER.sub(repl, expr)

    def _tree(self, sample: Dict[str, Any], systematic: Dict[str, Any], template: Optional[str]) -> str:
        tree = sample.get("Tree", "")
        if template is not None:
            tree = systematic.get(template, {}).get("Tree", tree)
        prefix = self.reg.samples[sample["Name"]].syst_tree_prefix()
        return tree.replace("{prefix}", prefix)

    def _paths(self, sample, systematic, template) -> Tuple[str, ...]:
        general = self.comp.config["General"]["InputPath"]
        sample_paths = sample.get("SamplePath")
        if template is not None:
            sample_paths = systematic.get(template, {}).get("SamplePath", sample_paths)
        if "{SamplePath}" in general:
            return tuple(general.replace("{SamplePath}", p) for p in cabinetry.configuration._setting_to_list(sample_paths))
        return (general,)

    # ------------------------------------------------------------------ planning
    def plan(self) -> None:
        if self._planned:
            return
        cabinetry.route.apply_to_all_templates(self.comp.config, self._plan_one)
        self._planned = True
        n_jobs = sum(len(g.jobs) for g in self.groups.values())
        log.info(
            f"planned {n_jobs} ntuple histograms from {len(self.groups)} (file, tree) combinations, "
            f"{len(self.post)} derived and {len(self.hist_jobs)} histogram-input templates"
        )

    def _plan_one(self, region, sample, systematic, template) -> None:
        name = cabinetry.histo.name(region, sample, systematic, template=template)
        sname = sample["Name"]
        self.sample_of[name] = sname
        systname = systematic.get("Name", "") if template is not None else ""
        custom = self.reg.custom.get((sname, systname, template))
        if custom is not None and custom.kind in ("yields", "scale"):
            self.post.append((name, region, sample, systematic, template, custom))
            return
        if custom is None and sname in {k[0] for k in self.reg.custom if k[1] == "" and k[2] is None}:
            # sample built from yields: only systematics with custom templates are possible
            raise ValueError(f"no template rule for {name}")
        if self.reg.input_type == "histogram":
            # everything that is not derived from other templates is read from the
            # histogram file (incl. event-subset theory variations)
            self.hist_jobs.append((name, region, sample, systematic, template))
            return
        entry = self.reg.regions[region["Name"]]
        variable = region["Variable"]
        selection = region.get("Filter", "") or ""
        if sample.get("Filter"):
            selection = sample["Filter"]
        weight_expr = None
        if template is not None:
            override = systematic.get(template, {})
            variable = override.get("Variable", variable)
            selection = override.get("Filter", selection)
            weight_expr = override.get("Weight")
        if custom is not None:  # kind == "ntuple": region dependent weight
            weight_expr = custom.weight(region["Name"])
        is_data = bool(sample.get("Data", False))
        weight = "1.0" if is_data else self._weight(sname, weight_expr)
        tree = self._tree(sample, systematic, template)
        paths = self._paths(sample, systematic, template)
        key = (paths, tree)
        group = self.groups.setdefault(key, TreeGroup(paths=paths, tree=tree))
        group.nominal |= template is None
        group.jobs.append(
            NtupleJob(
                name=name,
                sample=sname,
                systematic=systname,
                template=template,
                edges=tuple(region["Binning"]),
                variable=variable,
                selection=selection,
                weight=weight,
                underflow=bool(entry.underflow),
                overflow=bool(entry.overflow),
                nominal_name=cabinetry.histo.name(region, sample, {}, template=None),
                is_data=is_data,
            )
        )

    # ------------------------------------------------------------------ computing
    def compute(self, samples: Optional[Sequence[str]] = None) -> None:
        """Computes all histograms (or those of the given samples)."""
        self.plan()
        wanted = None if samples is None else set(samples)
        groups = []
        for group in self.groups.values():
            jobs = [j for j in group.jobs if (wanted is None or j.sample in wanted) and j.name not in self.store]
            if jobs:
                groups.append(TreeGroup(paths=group.paths, tree=group.tree, jobs=jobs, nominal=group.nominal))
        if groups:
            log.info(f"reading {len(groups)} (file, tree) combinations")
        results = self._run(groups)
        for group, res in zip(groups, results):
            if res.get("__missing__"):
                self._handle_missing(group)
                continue
            self.store.update(res)
        self._histogram_inputs(wanted)
        self._derived(wanted)
        self._resolve_fallbacks()

    def _run(self, groups: List[TreeGroup]) -> List[Dict[str, Any]]:
        if self.n_workers == 1 or len(groups) < 2:
            return [process_group(g) for g in groups]
        ctx = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(self.n_workers, mp_context=ctx) as pool:
            return list(pool.map(process_group, groups, chunksize=max(1, len(groups) // (4 * self.n_workers))))

    def _handle_missing(self, group: TreeGroup) -> None:
        nominal_jobs = [j for j in group.jobs if j.template is None]
        if nominal_jobs:
            raise RuntimeError(f"nominal tree '{group.tree}' not found in {list(group.paths)}")
        if self.reg.missing_tree == "error":
            raise RuntimeError(f"systematic tree '{group.tree}' not found in {list(group.paths)}")
        log.warning(f"tree '{group.tree}' not found in {list(group.paths)}: using the nominal (no effect)")
        for job in group.jobs:
            self.fallback_to_nominal[job.name] = job.nominal_name

    def _resolve_fallbacks(self) -> None:
        for name, nominal in list(self.fallback_to_nominal.items()):
            if nominal in self.store:
                values, variances = self.store[nominal]
                self.store[name] = (values.copy(), variances.copy())

    def _histogram_inputs(self, wanted) -> None:
        import uproot

        analysis = self.reg.analysis
        for name, region, sample, systematic, template in self.hist_jobs:
            if name in self.store or (wanted is not None and sample["Name"] not in wanted):
                continue
            entry = self.reg.regions[region["Name"]]
            edges = np.asarray(region["Binning"])
            files = self._paths(sample, systematic, template)
            hpath = analysis.histogram_path(
                region["Name"], sample["Name"], systematic.get("Name", "") if template else "", template
            )
            h = None
            for path in files:
                with uproot.open(path) as handle:
                    if hpath in handle:
                        h = handle[hpath]
                        values = np.asarray(h.values(flow=True), dtype=float)
                        variances = h.variances(flow=True)
                        variances = values.copy() if variances is None else np.asarray(variances, dtype=float)
                        hedges = np.asarray(h.axis().edges())
                        break
            if h is None:
                if template is not None and self.reg.missing_tree != "error":
                    log.warning(f"histogram {hpath} not found: using nominal")
                    self.fallback_to_nominal[name] = cabinetry.histo.name(region, sample, {}, template=None)
                    continue
                raise KeyError(f"histogram {hpath} not found in {list(files)}")
            inner_v, inner_e = values[1:-1], variances[1:-1]
            if region["Variable"] == CUTS:
                v, e = np.array([values.sum()]), np.array([variances.sum()])
            else:
                if len(hedges) != len(edges) or not np.allclose(hedges, edges):
                    raise ValueError(f"binning of {hpath} {hedges} differs from region binning {edges}")
                v, e = inner_v.copy(), inner_e.copy()
                if entry.underflow:
                    v[0] += values[0]
                    e[0] += variances[0]
                if entry.overflow:
                    v[-1] += values[-1]
                    e[-1] += variances[-1]
            self.store[name] = (v, e)

    def _derived(self, wanted) -> None:
        for name, region, sample, systematic, template, custom in self.post:
            if name in self.store or (wanted is not None and sample["Name"] not in wanted):
                continue
            nbins = len(region["Binning"]) - 1
            if custom.kind == "yields":
                values, variances = custom.yields(region["Name"], nbins)
            else:  # scale the nominal
                nominal = cabinetry.histo.name(region, sample, {}, template=None)
                if nominal not in self.store:
                    raise RuntimeError(f"nominal {nominal} needed for {name} not available")
                nom_v, nom_e = self.store[nominal]
                factors = np.asarray(custom.factors(region["Name"], nbins), dtype=float)
                values, variances = nom_v * factors, nom_e * factors**2
            self.store[name] = (np.asarray(values, dtype=float), np.asarray(variances, dtype=float))

    def release(self, samples: Sequence[str]) -> None:
        """Frees memory of histograms of the given samples."""
        drop = set(samples)
        for name in [n for n in self.store if self.sample_of.get(n) in drop]:
            del self.store[name]

    # ------------------------------------------------------------------ cabinetry
    def router(self) -> cabinetry.route.Router:
        router = cabinetry.route.Router()

        @router.register_template_builder(region_name="*", sample_name="*", systematic_name="*", template="*")
        def from_store(region, sample, systematic, template):
            name = cabinetry.histo.name(region, sample, systematic, template=template)
            if name not in self.store:
                raise KeyError(f"histogram {name} was not computed")
            values, variances = self.store[name]
            values = np.asarray(values, dtype=float).copy()
            if not sample.get("Data", False):
                bad = values < MIN_YIELD
                if np.any(values < -MIN_YIELD):
                    log.warning(f"{name}: negative yield(s) {values[values < 0]} set to {MIN_YIELD}")
                values[bad] = np.where(values[bad] < 0, MIN_YIELD, values[bad])
            hist = bh.Histogram(bh.axis.Variable(region["Binning"]), storage=bh.storage.Weight())
            view = hist.view()
            view.value = values
            view.variance = np.asarray(variances, dtype=float)
            return hist

        return router

    def build(self, config: Dict[str, Any]) -> None:
        """Runs cabinetry's template building + post-processing for ``config``."""
        samples = [s["Name"] for s in config["Samples"]]
        self.compute(samples)
        cabinetry.templates.build(config, router=self.router())
        cabinetry.templates.postprocess(config)
