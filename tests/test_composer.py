import pytest

from fit2l3l.workspace.composer import Composer


def _cfg(fit_type="bkg", regions=None, **general):
    g = {"Name": "t", "FitType": fit_type, "Systematics": "NONE"}
    g.update(general)
    cfg = {
        "General": g,
        "Regions": regions or {"CR": ["CRWZ2j"], "VR": ["VRWZ4j"], "SR": ["RpcSSwz1"]},
        "Samples": {"Data": "ntuple", "Backgrounds": ["WZ", "ttZ", "Fakes"]},
    }
    if fit_type == "excl":
        cfg["Samples"]["Signal"] = {"Grid": "Mc16SusyGG2StepWZ", "Points": ["376776"], "SigNameSchema": "SP:4,LSP:7"}
    return cfg


@pytest.mark.parametrize(
    "regions, fitted",
    [
        ({"SR": ["RpcSSwz1"]}, {"RpcSSwz1"}),  # SR used as CR
        ({"VR": ["VRWZ4j"]}, {"VRWZ4j"}),  # VR used as CR
        ({"CR": ["CRWZ2j"], "SR": ["RpcSSwz1"]}, {"CRWZ2j"}),  # SR -> VR
        ({"CR": ["CRWZ2j"], "VR": ["VRWZ4j"], "SR": ["RpcSSwz1"]}, {"CRWZ2j"}),
    ],
)
def test_bkg_only_roles(regions, fitted):
    comp = Composer(_cfg(regions=regions)).compose()
    assert {r["name"] for r in comp.meta["regions"] if r["fit"]} == fitted


def test_overflow_defaults():
    comp = Composer(_cfg(regions={"CR": ["CRWZ2j"], "SR": [{"Name": "RpcSSwz1", "Variable": "met"}, "RpcSSwz4"]})).compose()
    reg = comp.registry.regions
    assert reg["CRWZ2j"].overflow and reg["CRWZ2j"].underflow
    assert reg["RpcSSwz1"].overflow and not reg["RpcSSwz1"].underflow
    assert not reg["RpcSSwz4"].overflow  # single bin SR


def test_systematics_selection_and_flat():
    comp = Composer(_cfg(Systematics="EG,FLAT30")).compose()
    names = {s["Name"] for s in comp.config["Systematics"]}
    assert "EG_Scale_ALL" in names and "elID" in names
    assert "FlatSyst_WZ" in names and "FlatSyst_Fakes" not in names
    assert "JET_Flavor_Response" not in names
    flat = next(s for s in comp.config["Systematics"] if s["Name"] == "FlatSyst_WZ")
    assert flat["Up"]["Normalization"] == pytest.approx(0.3)


def test_ddflat_switches_on_data_driven():
    comp = Composer(_cfg(Systematics="DDFLAT20")).compose()
    names = {s["Name"] for s in comp.config["Systematics"]}
    assert any(n.startswith("syst_Fakes_") for n in names)
    assert any(n.startswith("stat_Fakes_") for n in names)


def test_normfactor_normalisation_and_lumi():
    cfg = _cfg(Systematics="EG,TH")
    cfg["NormFactors"] = {"WZ": "CRWZ2j"}
    comp = Composer(cfg).compose()
    assert "WZ" not in comp.registry.lumi_samples  # free floating: no lumi uncertainty
    assert "ttZ" in comp.registry.lumi_samples
    assert "Fakes" not in comp.registry.lumi_samples
    assert comp.registry.normalize_to["EG_Scale_ALL"]["WZ"] == ["CRWZ2j"]
    assert "ttZ" not in comp.registry.normalize_to["EG_Scale_ALL"]
    assert any(nf["Name"] == "mu_WZ" for nf in comp.config["NormFactors"])


def test_fixed_nf_and_extra_weight():
    cfg = _cfg()
    cfg["FixedNF"] = ["WZ"]
    cfg["ExtraSampleWeights"] = {"WZ": "weightSpecial"}
    comp = Composer(cfg).compose()
    wz = next(s for s in comp.config["Samples"] if s["Name"] == "WZ")
    assert "weightSpecial" in wz["Weight"] and "0.859955" in wz["Weight"]
    assert any(s["Name"] == "NF_WZ" for s in comp.config["Systematics"])


def test_discovery_single_bin_and_dummy_sample():
    cfg = _cfg("disc", regions={"SR": [{"Name": "RpcSSwz1", "Variable": "met"}]})
    comp = Composer(cfg).compose()
    assert comp.config["Regions"][0]["Variable"] == "cuts"
    assert any(s["Name"] == "DiscoveryMode_RpcSSwz1" for s in comp.config["Samples"])
    assert comp.config["General"]["POI"] == "mu_SIG"


def test_exclusion_signal_and_theory():
    comp = Composer(_cfg("excl", regions={"SR": ["RpcGGwz2"]}, Systematics="TH")).compose()
    assert comp.registry.signal_samples == ["signal376776"]
    sig = comp.config_for("signal376776")
    assert any(s["Name"] == "signal376776" for s in sig["Samples"])
    bkg = comp.config_for(None)
    assert not any(s["Name"].startswith("signal") for s in bkg["Samples"])
    assert bkg["General"]["POI"] == ""


def test_errors():
    with pytest.raises(ValueError):
        Composer(_cfg("excl", regions={"CR": ["CRWZ2j"]})).compose()  # no SR
    with pytest.raises(KeyError):
        Composer(_cfg(regions={"SR": ["NotARegion"]})).compose()
    with pytest.raises(ValueError):
        Composer(_cfg(Systematics="NOTAGROUP")).compose()
