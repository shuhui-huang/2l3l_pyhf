"""Shared fixtures: small toy ntuples and workspaces built from them."""

import pathlib

import pytest

from fit2l3l.utils import load_yaml, setup_logging

REPO = pathlib.Path(__file__).resolve().parent.parent
POINTS = ["376776", "376777", "376778", "376779", "376781", "376786"]


@pytest.fixture(scope="session")
def toy_dir(tmp_path_factory):
    from fit2l3l.cli import make_toy_inputs

    setup_logging("WARNING")
    out = tmp_path_factory.mktemp("toys")
    make_toy_inputs.main([
        "--output", str(out / "InputTrees"), "--events", "1500", "--signal-events", "800",
        "--grid", "Mc16SusyGG2StepWZ", "--schema", "SP:4,LSP:7", "--points", ",".join(POINTS),
        "--syst-groups", "EG,MET", "--seed", "7",
    ])
    return out


def _build(toy_dir, name, points=None, **overrides):
    from fit2l3l.workspace.builder import build_workspace

    cfg = load_yaml(REPO / "configs" / "workspace" / name)
    over = {"InputPath": str(toy_dir / "InputTrees"), "OutputDir": str(toy_dir / "results")}
    over.update(overrides)
    return build_workspace(cfg, overrides=over, points=points)


@pytest.fixture(scope="session")
def bkg_ws(toy_dir):
    return _build(toy_dir, "bkgonly_RpcSSwz.yaml", Systematics="EG,MET,FT,FAKES,CHFLIP,TH")


@pytest.fixture(scope="session")
def excl_ws(toy_dir):
    return _build(toy_dir, "excl_GG2StepWZ_RpcGGwz.yaml", points=POINTS, Systematics="EG,FAKES,TH")


@pytest.fixture(scope="session")
def disc_ws(toy_dir):
    return _build(toy_dir, "disc_RpcGGwz2.yaml", Systematics="FAKES")


@pytest.fixture(scope="session")
def excl_nosys_ws(toy_dir):
    return _build(toy_dir, "excl_GG2StepWZ_RpcGGwz.yaml", points=["376776", "376781"], Systematics="NONE",
                  Name="excl_nosys")
