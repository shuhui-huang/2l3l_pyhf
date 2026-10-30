"""Sample definitions of the SS/3L analysis (ported from ``fitConfigSS3L.py``).

Input ntuples follow the HistFitter conventions:

* MC backgrounds : ``<InputPath>/<sample>.<lumi>.root``, trees ``<sample>_nom`` and
  ``<sample>_<SYSTEMATIC>`` for kinematic systematics
* data           : ``<InputPath>/data.<lumi>.root``, tree ``data_nom``
* signal         : ``<InputPath>/signal.<lumi>.root``, trees ``signal<DSID>_nom`` ...
  (see ``signals.py``)

Data-driven backgrounds (fakes, charge flip) are not read from ntuples: their
yields, statistical and systematic uncertainties come from
``share/hardCodedYields.json`` (keys given by :class:`DataDrivenYields`).

Which of these samples enter a workspace is chosen in the YAML
(``Samples.Backgrounds``).
"""

from fit2l3l.analysis.base import DATA, DATADRIVEN, DataDrivenYields, Sample, WeightSyst

# HistFitter ROOT colours translated to hex
_COLORS = {
    "kCyan+1": "#00b7c3",
    "kOrange": "#ffcc00",
    "kGreen+2": "#00a651",
    "kViolet-9": "#c7a5f0",
    "kAzure+7": "#3e9cde",
    "kOrange+1": "#ff9f1c",
    "kRed+2": "#a3121b",
    "kOrange-9": "#ffd9a8",
}


def build_samples():
    """Returns {name: Sample} for all samples known to the analysis."""
    samples = [
        # ---- data ---------------------------------------------------------
        Sample("data", kind=DATA, color="#000000", label="Data", file="data.{lumi}.root", tree="data_nom"),
        # ---- irreducible backgrounds (MC) ---------------------------------
        Sample("Multitop", color=_COLORS["kCyan+1"], label="OtherTop(s)"),
        Sample("SM4tops", color=_COLORS["kOrange"], label="SM 4tops"),
        Sample("OtherMultiboson", color=_COLORS["kGreen+2"], label="WW, ZZ, VVV"),
        Sample("ttW", color=_COLORS["kViolet-9"], label=r"$t\bar{t}W$"),
        Sample("ttZ", color=_COLORS["kAzure+7"], label=r"$t\bar{t}Z$"),
        Sample("WZ", color=_COLORS["kOrange+1"], label="WZ"),
        # older splittings (kept for tables / legacy configurations)
        Sample("ttH", color=_COLORS["kOrange"], label=r"$t\bar{t}H$"),
        Sample("RareTop_NottH", color=_COLORS["kCyan+1"], label=r"$tWZ$, $t\bar{t}VV$, 3t, 4t"),
        Sample("Multiboson", color=_COLORS["kGreen+2"], label="Multiboson"),
        Sample("ttV", color=_COLORS["kViolet-9"], label=r"$t\bar{t}V$"),
        # ---- fakes estimated with MC templates ----------------------------
        # (do not use together with the data-driven Fakes/MisCharge)
        Sample(
            "TTbarSgTop",
            color=_COLORS["kRed+2"],
            label=r"$t\bar{t}$ & single top",
            weights=["totweight", "lumiScaling"],
            experimental_systematics="none",
            extra_systematics=[
                WeightSyst("MCtemplate", up="mctemplate_up", down="mctemplate_dw", replace=None, group="MCTEMPLATE")
            ],
        ),
        Sample(
            "VJets",
            color=_COLORS["kOrange-9"],
            label="V+jets",
            weights=["totweight", "lumiScaling"],
            experimental_systematics="none",
            extra_systematics=[
                WeightSyst("MCtemplate", up="mctemplate_up", down="mctemplate_dw", replace=None, group="MCTEMPLATE")
            ],
        ),
        # ---- data-driven backgrounds (hard-coded yields) -------------------
        Sample(
            "MisCharge",
            kind=DATADRIVEN,
            color=_COLORS["kRed+2"],
            label="Charge-flip",
            staterror=False,
            norm_by_theory=False,
            yields=DataDrivenYields(
                nominal="mischCount",
                stat="mischStatErr",
                syst_up="mischSystErrUp",
                syst_down="mischSystErrDown",
                group="CHFLIP",
            ),
        ),
        Sample(
            "Fakes",
            kind=DATADRIVEN,
            color=_COLORS["kOrange-9"],
            label="Fake/non-prompt",
            staterror=False,
            norm_by_theory=False,
            yields=DataDrivenYields(
                nominal="fakeCount",
                stat="fakeStatErr",
                syst_up="fakeSystErrUp",
                syst_down="fakeSystErrDown",
                group="FAKES",
            ),
        ),
    ]
    # MC samples with weights=None use the analysis default (NOMINAL_WEIGHTS)
    return {s.name: s for s in samples}


def histogram_path(region, sample, systematic, template):
    """Path of a histogram inside the input file for ``InputType: histogram``.

    Convention: ``<region>/<sample>`` for the nominal and
    ``<region>/<sample>_<systematic>_<Up|Down>`` for variations.  Adapt to the
    naming of your histogram files.  ``systematic`` is the cabinetry systematic
    name, ``template`` is ``"Up"``, ``"Down"`` or None.
    """
    path = f"{region}/{sample}"
    if template is not None:
        path += f"_{systematic}_{template}"
    return path
