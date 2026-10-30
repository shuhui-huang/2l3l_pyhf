"""Regression tests for issues found in the code review."""

import json

import numpy as np
import pytest

from fit2l3l.analysis.base import Binning
from fit2l3l.workspace.composer import Composer
from fit2l3l.utils import read_json


def test_binning_tuple_vs_edges():
    assert Binning.from_any((3, 0.0, 300.0)).n == 3
    edges = Binning.from_any([100e3, 200e3, 400e3])
    assert edges.n == 2 and edges.bin_edges.tolist() == [100e3, 200e3, 400e3]
    with pytest.raises(ValueError):
        Binning.from_any([3, 2, 1])


def _cfg(**general):
    g = {"Name": "t", "FitType": "bkg", "Systematics": "FAKES"}
    g.update(general)
    return {
        "General": g,
        "Regions": {"VR": [{"Name": "VRWZ4j", "Variable": "met"}]},
        "Samples": {"Data": "ntuple", "Backgrounds": ["WZ", "Fakes"]},
    }


def test_dd_stat_no_effect_in_empty_bins():
    comp = Composer(_cfg()).compose()
    up = comp.registry.custom[("Fakes", "stat_Fakes_VRWZ4j", "Up")].yields("VRWZ4j", 5)[0]
    nominal = comp.registry.custom[("Fakes", "", None)].yields("VRWZ4j", 5)[0]
    empty = nominal == 0
    assert empty.any()
    assert np.all(up[empty] == 0) and np.all(up[~empty] > nominal[~empty])


def test_missing_dd_yields_error_or_zero():
    cfg = _cfg()
    cfg["Regions"] = {"VR": ["RpcGGwz1_noMet"]}  # no fakes entry in the JSON
    comp = Composer(cfg).compose()
    with pytest.raises(ValueError):
        comp.registry.custom[("Fakes", "", None)].yields("RpcGGwz1_noMet", 1)
    comp = Composer(dict(cfg, General=dict(cfg["General"], MissingYields="zero"))).compose()
    assert comp.registry.custom[("Fakes", "", None)].yields("RpcGGwz1_noMet", 1)[0].tolist() == [0.0]


def test_variable_alias_in_yields_json():
    cfg = _cfg()
    cfg["Regions"] = {"VR": [{"Name": "RpcGGslep1_noMetOverHtlep", "Variable": "met/ht_lept"}]}
    comp = Composer(cfg).compose()  # stored as "metht_lept" in the JSON
    values = comp.registry.custom[("Fakes", "", None)].yields("RpcGGslep1_noMetOverHtlep", 4)[0]
    assert len(values) == 4


def test_fixed_nf_validation():
    cfg = _cfg()
    cfg["FixedNF"] = ["Fakes"]
    with pytest.raises(ValueError):
        Composer(cfg).compose()
    cfg["FixedNF"] = ["NotASample"]
    with pytest.raises(ValueError):
        Composer(cfg).compose()


def test_extra_weight_with_comma():
    cfg = _cfg()
    cfg["ExtraSampleWeights"] = {"WZ": "max(w1, w2)"}
    comp = Composer(cfg).compose()
    assert "max(w1, w2)" in comp.registry.sample_weights["WZ"]


def test_condor_seed_order(tmp_path):
    from fit2l3l.fitting import batch

    sub = batch.write_jobs("fit.yaml", str(tmp_path), ["p1", "p2", "p3"], points_per_job=2, seeds=[7, 8],
                           system="condor", directory=str(tmp_path / "b"))
    text = sub.read_text()
    assert "queue seed, points from" in text
    lines = (tmp_path / "b" / "jobs_condor.txt").read_text().splitlines()
    assert lines[0] == "7 p1,p2"


def test_discovery_table_excludes_signal(disc_ws, tmp_path):
    from fit2l3l.fitting.runner import run_fits

    cfg = {"General": {"Workspace": str(disc_ws["workspace"].parent), "OutputDir": str(tmp_path)},
           "Fits": [{"Name": "d", "Plots": [], "Tables": ["yields", "systematics"], "UpperLimit": {"Enabled": False}}]}
    run_fits(cfg)
    table = read_json(tmp_path / "d" / "tables" / "yields_fit_regions.json")
    syst = read_json(tmp_path / "d" / "tables" / "systematics_postfit.json")
    sr = table["channels"][0]
    total, err = table["rows"][sr]["fitted"]["total"]
    assert "DiscoveryMode_RpcGGwz2" not in table["samples"]
    assert err == pytest.approx(syst[sr]["total_uncertainty"], rel=0.05)


def test_exclusion_toys_with_theory_and_limits(excl_nosys_ws, tmp_path):
    from fit2l3l.fitting import exclusion
    from fit2l3l.fitting.common import load_context
    from fit2l3l.fitting.runner import run_fits

    base = {"Name": "t", "Calculator": "toybased", "NToys": 40, "SaveToys": True, "Points": ["376781"],
            "UpperLimit": {"Enabled": True, "Scan": [0, 8, 5]}}
    for seed in (1, 2):
        cfg = {"General": {"Workspace": str(excl_nosys_ws["workspace"].parent), "OutputDir": str(tmp_path)},
               "Fits": [dict(base, Seed=seed)]}
        run_fits(cfg, merge=False)
    points = exclusion.merge(tmp_path / "t", load_context(excl_nosys_ws["workspace"].parent))
    p = points[0]
    assert p["toy_files"] == 2 and p["ntoys"] == 80
    assert p["upperLimitMethod"] == "toys" and p["upperLimit"] > 0
    assert "CLs_theory_up" in p and "CLs_theory_down" in p


def test_missing_signal_tree_is_skipped(toy_dir):
    from conftest import REPO

    from fit2l3l.utils import load_yaml
    from fit2l3l.workspace.builder import build_workspace

    cfg = load_yaml(REPO / "configs" / "workspace" / "excl_GG2StepWZ_RpcGGwz.yaml")
    over = {"InputPath": str(toy_dir / "InputTrees"), "OutputDir": str(toy_dir / "results"),
            "Systematics": "NONE", "Name": "excl_skip"}
    out = build_workspace(cfg, overrides=over, points=["376776", "376800"])  # 376800 has no trees
    patchset = read_json(out["patchset"])
    assert [p["metadata"]["name"] for p in patchset["patches"]] == ["signal376776"]
    assert "signal376800" in (out["workspace"].parent / "skipped_points.txt").read_text()
