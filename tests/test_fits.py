import json

import numpy as np
import pytest

from fit2l3l.fitting.runner import run_fits
from fit2l3l.utils import read_json


def test_bkg_only_fit(bkg_ws, tmp_path):
    cfg = {
        "General": {"Workspace": str(bkg_ws["workspace"].parent), "OutputDir": str(tmp_path)},
        "Fits": [{"Name": "bkg", "Plots": ["postfit", "summary"], "Tables": ["yields", "systematics"]}],
    }
    res = run_fits(cfg)["bkg"]
    assert res["fit_regions"] == ["CRWZ2j"]
    mu, err = res["normfactors"]["mu_WZ"]
    assert 0.5 < mu < 1.5 and 0 < err < 0.5
    table = read_json(tmp_path / "bkg" / "tables" / "yields_validation_regions.json")
    assert table["rows"]["RpcSSwz1"]["observed"] is None  # blinded
    syst = read_json(tmp_path / "bkg" / "tables" / "systematics_postfit.json")
    assert syst["CRWZ2j"]["total_uncertainty"] > 0
    assert (tmp_path / "bkg" / "plots" / "summary_postfit.pdf").exists()


def test_discovery_fit(disc_ws, tmp_path):
    cfg = {
        "General": {"Workspace": str(disc_ws["workspace"].parent), "OutputDir": str(tmp_path)},
        "Fits": [{"Name": "disc", "Plots": [], "Tables": [], "UpperLimit": {"Scan": [0, 60, 13]}}],
    }
    res = run_fits(cfg)["disc"]
    assert 0 < res["p0"] <= 0.5
    assert res["S95_exp_band"][0] < res["S95_exp"] < res["S95_exp_band"][4]
    assert res["S95_obs"] > 0
    assert (tmp_path / "disc" / "upper_limit_table.tex").exists()


def test_exclusion_and_contours(excl_ws, tmp_path):
    cfg = {
        "General": {"Workspace": str(excl_ws["workspace"].parent), "OutputDir": str(tmp_path)},
        "Fits": [{"Name": "excl", "UpperLimit": {"Enabled": True, "NPoints": 6}, "Points": ["376776", "376781"]}],
    }
    res = run_fits(cfg)["excl"]
    assert not res["failed"]
    points = read_json(tmp_path / "excl" / "hypotest_list.json")
    assert len(points) == 2
    for p in points:
        assert 0 <= p["clsd2s"] <= p["CLsexp"] <= p["clsu2s"] <= 1
        assert p["expectedUpperLimitMinus1Sig"] <= p["expectedUpperLimit"] <= p["expectedUpperLimitPlus1Sig"]
        # CLs at mu = upper limit is ~0.05 by construction: CLs(mu=1) < 0.05 <=> UL < 1
        assert (p["CLs"] < 0.05) == (p["upperLimit"] < 1.0)
        assert p["CLs_theory_up"] <= p["CLs"] <= p["CLs_theory_down"]
    missing = (tmp_path / "excl" / "missing_points.txt").read_text().split()
    assert len(missing) == 4


def test_contour_plot(tmp_path):
    from fit2l3l.plotting import contours

    rng = np.random.default_rng(1)
    pts = []
    for m0 in np.arange(1000, 2001, 100):
        for m12 in np.arange(100, m0 - 300, 200):
            z = 4.0 - (m0 - 1000) / 150.0 - m12 / 600.0
            cls = float(contours.norm.sf(z))
            pts.append({"name": f"p{m0}_{m12}", "m0": float(m0), "m12": float(m12), "masses": [m0, m12], "CLs": cls,
                        "CLsexp": cls, "clsd1s": cls / 2, "clsu1s": min(1, cls * 2), "clsd2s": cls / 4, "clsu2s": min(1, cls * 4)})
    out = contours.plot_contours(pts, {"x_range": (1000, 2000), "y_range": (0, 1800)}, tmp_path / "c")
    assert out["lines"]["observed"] >= 1
    best = contours.best_of({"A": pts, "B": [dict(p, CLsexp=p["CLsexp"] * 0.5) for p in pts]})
    assert all(p["bestSR"] == "B" for p in best)
