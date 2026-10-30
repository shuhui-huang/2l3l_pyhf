import json

import numpy as np
import pyhf

from fit2l3l.utils import read_json, share_path


def _samples(ws):
    return {(c["name"], s["name"]): s for c in ws["channels"] for s in c["samples"]}


def test_bkg_workspace(bkg_ws):
    ws = json.loads(bkg_ws["workspace"].read_text())
    meta = read_json(bkg_ws["meta"])
    pyhf.Workspace(ws)
    samples = _samples(ws)
    # WZ is normalised in CRWZ2j: its systematics have no normalisation effect there
    wz_cr = samples[("CRWZ2j", "WZ")]
    assert not [m for m in wz_cr["modifiers"] if m["type"] == "normsys"]
    assert any(m["type"] == "normfactor" and m["name"] == "mu_WZ" for m in wz_cr["modifiers"])
    assert not any(m["type"] == "lumi" for m in wz_cr["modifiers"])
    assert any(m["type"] == "lumi" for m in samples[("CRWZ2j", "ttZ")]["modifiers"])
    params = {p["name"]: p for p in ws["measurements"][0]["config"]["parameters"]}
    assert params["lumi"]["sigmas"] == [0.017]
    # data-driven yields from the JSON
    yields = read_json(share_path("hardCodedYields.json"))
    assert np.isclose(samples[("CRWZ2j", "Fakes")]["data"][0], yields["fakeCount"]["CRWZ2j"]["cuts"][0])
    assert np.allclose(samples[("RpcSSwz1", "Fakes")]["data"], yields["fakeCount"]["RpcSSwz1"]["met"], atol=1e-5)
    # SRs are blinded: data = background expectation
    for channel in ws["channels"]:
        obs = next(o["data"] for o in ws["observations"] if o["name"] == channel["name"])
        total = np.sum([s["data"] for s in channel["samples"]], axis=0)
        if channel["name"] in meta["blinded_regions"]:
            assert np.allclose(obs, total)
        else:
            assert np.allclose(obs, np.round(obs))


def test_exclusion_patchset(excl_ws):
    ws = pyhf.Workspace(json.loads(excl_ws["workspace"].read_text()))
    patchset = pyhf.PatchSet(read_json(excl_ws["patchset"]))
    assert len(patchset.patches) == 6
    patched = patchset.apply(ws, patchset.patches[0].name)
    model = patched.model()
    assert model.config.poi_name == "mu_SIG"
    assert patchset.patches[0].name in model.config.samples
    sig = next(s for c in patched["channels"] for s in c["samples"] if s["name"] == patchset.patches[0].name)
    assert {m["type"] for m in sig["modifiers"]} >= {"normfactor", "lumi", "staterror"}


def test_discovery_workspace(disc_ws):
    ws = json.loads(disc_ws["workspace"].read_text())
    dummy = [s for c in ws["channels"] for s in c["samples"] if s["name"].startswith("DiscoveryMode_")]
    assert len(dummy) == 1 and dummy[0]["data"] == [1.0]
    assert [m["type"] for m in dummy[0]["modifiers"]] == ["normfactor"]
