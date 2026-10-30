"""Creates toy ntuples in the HistFitter input format of the SS/3L analysis.

Only meant for testing the framework / trying the workflow without real inputs:
the kinematics are random numbers with roughly sensible shapes, the yields are
arbitrary.  Files written to ``--output`` (default ``InputTrees``)::

    <sample>.<lumi>.root : trees <sample>_nom, <sample>_<SYST> (kinematic systematics)
    data.<lumi>.root     : tree data_nom (pseudo-data drawn from the backgrounds)
    signal.<lumi>.root   : trees signal<DSID>_nom, signal<DSID>_<SYST>

Example::

    fit2l3l-toy-inputs --output InputTrees --grid Mc16SusyGG2StepWZ --points 376776,376777
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Dict, List, Optional

import numpy as np

from fit2l3l.analysis.base import TreeSyst, WeightSyst
from fit2l3l.utils import as_list, load_analysis, setup_logging
from fit2l3l.workspace import expressions

log = logging.getLogger("fit2l3l.toys")

# relative normalisation (expected yield after the SS/3L pre-selection) of the toy samples;
# scaled such that e.g. CRWZ2j has O(3000) events like the real analysis (the
# data-driven backgrounds come from hardCodedYields.json with real yields)
TOY_SCALE = 18.0
TOY_YIELDS = {
    "Multitop": 60.0,
    "SM4tops": 15.0,
    "OtherMultiboson": 150.0,
    "ttW": 220.0,
    "ttZ": 200.0,
    "WZ": 900.0,
    "ttH": 80.0,
    "RareTop_NottH": 40.0,
    "Multiboson": 300.0,
    "ttV": 400.0,
    "TTbarSgTop": 300.0,
    "VJets": 120.0,
}

# characteristics: (mean number of b-jets, mean jets, fraction on Z peak, met scale [GeV], leptons p3)
TOY_SHAPES = {
    "Multitop": (1.6, 5.0, 0.2, 90, 0.35),
    "SM4tops": (2.8, 7.5, 0.1, 110, 0.35),
    "OtherMultiboson": (0.2, 2.2, 0.5, 70, 0.4),
    "ttW": (1.4, 4.0, 0.0, 90, 0.25),
    "ttZ": (1.5, 4.5, 0.75, 85, 0.55),
    "WZ": (0.15, 2.5, 0.85, 75, 0.85),
    "ttH": (1.6, 5.0, 0.05, 90, 0.3),
    "RareTop_NottH": (1.3, 4.5, 0.3, 90, 0.4),
    "Multiboson": (0.2, 2.3, 0.6, 70, 0.6),
    "ttV": (1.5, 4.2, 0.4, 90, 0.4),
    "TTbarSgTop": (1.2, 3.5, 0.0, 70, 0.1),
    "VJets": (0.2, 2.0, 0.3, 50, 0.1),
}


def _kinematics(rng, n, shape, signal_scale: Optional[float] = None) -> Dict[str, np.ndarray]:
    nb_mean, nj_mean, zfrac, met_scale, p3 = shape
    if signal_scale is not None:
        nj_mean, met_scale = 6.5, signal_scale * 0.25
    nlep = np.where(rng.random(n) < p3, 3, 2)
    nlep = np.where(rng.random(n) < 0.03, 4, nlep)
    njets = rng.poisson(nj_mean, n) + 1
    nj40 = rng.binomial(njets, 0.75)
    nj50 = rng.binomial(nj40, 0.8)
    nb = np.minimum(rng.poisson(nb_mean, n), njets)
    gev = 1000.0
    met = (30 + rng.exponential(met_scale, n)) * gev
    pt_jets = rng.exponential(60, (n, 3)) + 25
    ht_jet = (njets * (45 + rng.exponential(45, n)) + pt_jets[:, 0]) * gev
    if signal_scale is not None:
        ht_jet = ht_jet + rng.normal(signal_scale * 0.6, signal_scale * 0.15, n).clip(0) * gev
    pt_l = np.sort(rng.exponential(45, (n, 3)) + 20, axis=1)[:, ::-1] * gev
    ht_lept = pt_l[:, 0] + pt_l[:, 1] + np.where(nlep >= 3, pt_l[:, 2], 0.0)
    meff = met + ht_jet + ht_lept
    onz = rng.random(n) < zfrac
    msfos = np.where(onz, rng.normal(91.2, 3.0, n), rng.uniform(12, 250, n)) * gev
    ht_bjet = ht_jet * np.clip(nb / np.maximum(njets, 1) + rng.normal(0, 0.05, n), 0, 1)
    return {
        "nSigLep": nlep.astype(np.int32),
        "NlepBL": nlep.astype(np.int32),
        "nBJets20": nb.astype(np.int32),
        "nJets25": njets.astype(np.int32),
        "nJets40": nj40.astype(np.int32),
        "nJets50": nj50.astype(np.int32),
        "met": met.astype(np.float32),
        "meff": meff.astype(np.float32),
        "ht_jet": ht_jet.astype(np.float32),
        "ht_lept": ht_lept.astype(np.float32),
        "ht_bjet": ht_bjet.astype(np.float32),
        "mSFOS": msfos.astype(np.float32),
        "mllSS": (rng.uniform(12, 300, n) * gev).astype(np.float32),
        "Pt_l": pt_l[:, 0].astype(np.float32),
        "Pt_subl": pt_l[:, 1].astype(np.float32),
        "Pt_thirdl": np.where(nlep >= 3, pt_l[:, 2], 0.0).astype(np.float32),
        "Pt_jet": (pt_jets[:, 0] * gev + 40 * gev).astype(np.float32),
        "Pt_subjet": (pt_jets[:, 1] * gev + 30 * gev).astype(np.float32),
        "dRl1j": rng.uniform(0.4, 4, n).astype(np.float32),
        "dRl2j": rng.uniform(0.4, 4, n).astype(np.float32),
        "dRSS": rng.uniform(0.2, 4.5, n).astype(np.float32),
        "dRll": rng.uniform(0.1, 4.5, n).astype(np.float32),
        "dPhiLLmet": rng.uniform(0, np.pi, n).astype(np.float32),
        "met_Sig": (met / gev / rng.uniform(5, 15, n)).astype(np.float32),
        "isSS15": (rng.random(n) < 0.9).astype(np.int32),
        "isSS30": (rng.random(n) < 0.8).astype(np.int32),
        "SSChannel": rng.integers(1, 4, n).astype(np.int32),
    }


def _weights(rng, n, norm, weight_names: List[str]) -> Dict[str, np.ndarray]:
    out = {}
    for name in weight_names:
        if name == "lumiScaling":
            out[name] = np.full(n, norm, dtype=np.float32)
        elif name in ("lumiScaling_up", "lumiScaling_down"):
            out[name] = np.full(n, norm * (1.1 if name.endswith("up") else 0.9), dtype=np.float32)
        elif name.endswith(("_up", "_dw", "_down", "Weight_up", "Weight_down")):
            sign = 1.0 if name.endswith("up") else -1.0
            size = rng.uniform(0.005, 0.04)
            out[name] = (1.0 + sign * size + rng.normal(0, 0.01, n)).astype(np.float32)
        else:
            out[name] = (1.0 + rng.normal(0, 0.02, n)).astype(np.float32)
    if "mcweight" in out:
        out["mcweight"] = np.where(rng.random(n) < 0.02, -1.0, 1.0).astype(np.float32)
    return out


def _vary(arrays: Dict[str, np.ndarray], rng, syst: str, direction: float) -> Dict[str, np.ndarray]:
    """Kinematic variation for a systematic tree (deterministic per systematic name)."""
    out = dict(arrays)
    seed = sum(map(ord, syst)) % 1000
    size = 0.01 + 0.03 * ((seed % 97) / 97.0)
    if syst.startswith(("JET", "EG", "MUON", "MET")):
        n = len(arrays["met"])
        smear = rng.normal(0, size / 2, n) if direction == 0 else np.full(n, direction * size)
        for key in ("met", "meff", "ht_jet"):
            out[key] = (arrays[key] * (1 + smear)).astype(np.float32)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="InputTrees")
    parser.add_argument("--lumi", default="138965")
    parser.add_argument("--analysis", default="ss3l")
    parser.add_argument("--samples", default="Multitop,SM4tops,OtherMultiboson,ttW,ttZ,WZ")
    parser.add_argument("--events", type=int, default=5000, help="events per background sample")
    parser.add_argument("--signal-events", type=int, default=3000)
    parser.add_argument("--grid", default=None, help="signal grid (share/susyGridFiles)")
    parser.add_argument("--points", default=None, help="comma separated DSIDs (default: all of the grid)")
    parser.add_argument("--schema", default="SP:4,LSP:5")
    parser.add_argument("--syst-groups", default="ALL", help="groups for which kinematic trees are written")
    parser.add_argument("--signal-strength", type=float, default=0.0, help="inject signal (first point) into data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    import uproot

    rng = np.random.default_rng(args.seed)
    analysis = load_analysis(args.analysis)
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    groups = analysis.all_groups if args.syst_groups.upper() == "ALL" else as_list(args.syst_groups)
    systs = analysis.systematics_for_groups([g for g in groups if g in analysis.systematic_groups])
    tree_systs = [s for s in systs if isinstance(s, TreeSyst)]
    weight_names = set(analysis.nominal_weights) | {"MCId", "totweight", "lumiScaling_up", "lumiScaling_down"}
    for s in analysis.systematics_for_groups(list(analysis.systematic_groups)):
        if isinstance(s, WeightSyst):
            for w in (s.up, s.down):
                if w:
                    weight_names |= expressions.free_symbols(w)
    for sample in analysis.samples.values():
        for s in sample.extra_systematics:
            if isinstance(s, WeightSyst):
                for w in (s.up, s.down):
                    if w:
                        weight_names |= expressions.free_symbols(w)
    weight_names |= {"wSystFactorLxyWeight_up", "wSystFactorLxyWeight_down"}
    kin_names = set(_kinematics(rng, 1, TOY_SHAPES["WZ"]))
    weight_names = sorted(weight_names - kin_names - {"MCId"})

    data_parts = []
    lumi = args.lumi
    for name in as_list(args.samples):
        n = args.events
        shape = TOY_SHAPES.get(name, (0.8, 3.5, 0.3, 80, 0.4))
        kin = _kinematics(rng, n, shape)
        norm = TOY_SCALE * TOY_YIELDS.get(name, 100.0) / n
        w = _weights(rng, n, norm, weight_names)
        mcid = np.full(n, 400000 + sum(map(ord, name)) % 1000, dtype=np.int32)
        if name == "OtherMultiboson":
            mcid = np.where(rng.random(n) < 0.3, rng.choice([364286, 364287], n), mcid).astype(np.int32)
        if name == "SM4tops":
            mcid = np.full(n, 412043, dtype=np.int32)
        nominal = {**kin, **w, "MCId": mcid, "totweight": np.ones(n, dtype=np.float32)}
        path = out / f"{name}.{lumi}.root"
        with uproot.recreate(path) as handle:
            handle[f"{name}_nom"] = nominal
            for syst in tree_systs:
                handle[f"{name}_{syst.up}"] = _vary(nominal, rng, syst.up, 0.0 if syst.down is None else +1.0)
                if syst.down is not None:
                    handle[f"{name}_{syst.down}"] = _vary(nominal, rng, syst.down, -1.0)
        log.info(f"wrote {path}")
        # pseudo-data: Poisson number of events drawn from the sample
        n_data = rng.poisson(TOY_SCALE * TOY_YIELDS.get(name, 100.0))
        idx = rng.choice(n, n_data, replace=n_data > n)
        data_parts.append({k: v[idx] for k, v in kin.items()})

    signal_data = None
    if args.grid:
        grid = analysis.signal_grid(args.grid, args.schema)
        points = grid.points if not args.points else [grid.point(p) for p in as_list(args.points)]
        path = out / f"signal.{lumi}.root"
        with uproot.recreate(path) as handle:
            for i, point in enumerate(points):
                n = args.signal_events
                kin = _kinematics(rng, n, TOY_SHAPES["WZ"], signal_scale=float(point.m0 - 0.7 * point.m12))
                # signal yield falls with the mass, harder spectrum for large mass splittings
                yield_ = 300.0 * np.exp(-(point.m0 - 1000.0) / 200.0)
                norm = yield_ / n
                w = _weights(rng, n, norm, weight_names)
                tree_name = f"signal{point.dsid}"
                nominal = {**kin, **w, "MCId": np.full(n, int(point.dsid), dtype=np.int32),
                           "totweight": np.ones(n, dtype=np.float32)}
                for extra in point.extra_weights:
                    nominal[extra] = np.ones(n, dtype=np.float32)
                handle[f"{tree_name}_nom"] = nominal
                for syst in tree_systs:
                    handle[f"{tree_name}_{syst.up}"] = _vary(nominal, rng, syst.up, 0.0 if syst.down is None else +1.0)
                    if syst.down is not None:
                        handle[f"{tree_name}_{syst.down}"] = _vary(nominal, rng, syst.down, -1.0)
                if i == 0 and args.signal_strength > 0:
                    n_sig = rng.poisson(args.signal_strength * yield_)
                    idx = rng.choice(n, min(n, n_sig), replace=False)
                    signal_data = {k: v[idx] for k, v in kin.items()}
        log.info(f"wrote {path} with {len(points)} signal points")

    if signal_data is not None:
        data_parts.append(signal_data)
    data = {k: np.concatenate([p[k] for p in data_parts]) for k in data_parts[0]}
    with uproot.recreate(out / f"data.{lumi}.root") as handle:
        handle["data_nom"] = data
    log.info(f"wrote {out / f'data.{lumi}.root'} ({len(data['met'])} events)")


if __name__ == "__main__":
    main()
