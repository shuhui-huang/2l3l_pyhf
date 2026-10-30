"""Systematic uncertainties of the SS/3L analysis (ported from ``fitConfigSS3L.py``).

Experimental systematics are organised in *groups*; a workspace YAML switches
groups on with ``General.Systematics`` (``ALL`` = :data:`ALL_GROUPS`).

* tree-based  : ``TreeSyst(name, up_tree_suffix, down_tree_suffix)``; the trees are
  ``<sample>_<suffix>`` (HistFitter convention).  ``down=None`` -> one-sided,
  symmetrised (HistFitter ``histoSysOneSideSym``).
* weight-based: ``WeightSyst(name, up, down, replace=<nominal factor>)`` replaces one
  factor of the nominal weight (HistFitter ``replaceElementList``), or multiplies
  by an extra factor when ``replace`` is None.

Groups ``FAKES``/``CHFLIP`` switch on the uncertainties of the data-driven
backgrounds (their yields/uncertainties live in ``share/hardCodedYields.json``),
``TH`` the theory uncertainties of :data:`THEORY_SOURCES`.

HistFitter ``histoSys``/``overallSys`` -> cabinetry ``NormPlusShape`` (a
``normsys`` + ``histosys`` pair in pyhf).  For backgrounds normalised in a control
region (``NormFactors`` in the YAML) the variations are re-normalised in the
control region(s) like HistFitter's ``overallNormSys``/``normHistoSys``.
"""

from fit2l3l.analysis.base import TheorySyst, TreeSyst, WeightSyst

#: nominal event weight (product of branches); ``lumiScaling`` normalises to Lumi
NOMINAL_WEIGHTS = [
    "wmu_nom",
    "wel_nom",
    "wtrig_nom",
    "wjet_nom",
    "mcweight",
    "wpu_nom",
    "mccampaignweight",
    "lumiScaling",
]

#: groups switched on by "ALL" (HistFitter: EG,JET,MU,MET,FT,PU,TRIG,FAKES,CHFLIP,TH)
ALL_GROUPS = ["EG", "JET", "MU", "MET", "FT", "PU", "TRIG", "FAKES", "CHFLIP", "TH"]


def _jet(name):
    return TreeSyst(name, f"{name}__1up", f"{name}__1down")


TREE_SYSTEMATICS = {
    "EG": [
        TreeSyst("EG_Scale_ALL", "EG_SCALE_ALL__1up", "EG_SCALE_ALL__1down"),
        TreeSyst("EG_Scale_AF2", "EG_SCALE_AF2__1up", "EG_SCALE_AF2__1down"),
        TreeSyst("EG_Resolution", "EG_RESOLUTION_ALL__1up", "EG_RESOLUTION_ALL__1down"),
    ],
    "JET": [
        _jet(n)
        for n in [
            "JET_EtaIntercalibration_Modelling",
            "JET_EtaIntercalibration_NonClosure_2018data",
            "JET_EtaIntercalibration_NonClosure_highE",
            "JET_EtaIntercalibration_NonClosure_negEta",
            "JET_EtaIntercalibration_NonClosure_posEta",
            "JET_EtaIntercalibration_TotalStat",
            "JET_Flavor_Composition",
            "JET_Flavor_Response",
            # "JET_PunchThrough_MC16",
            "JET_RelativeNonClosure_AFII",
            "JET_BJES_Response",
            "JET_EffectiveNP_Detector1",
            "JET_EffectiveNP_Detector2",
            "JET_EffectiveNP_Mixed1",
            "JET_EffectiveNP_Mixed2",
            "JET_EffectiveNP_Mixed3",
            "JET_EffectiveNP_Modelling1",
            "JET_EffectiveNP_Modelling2",
            "JET_EffectiveNP_Modelling3",
            "JET_EffectiveNP_Modelling4",
            "JET_EffectiveNP_Statistical1",
            "JET_EffectiveNP_Statistical2",
            "JET_EffectiveNP_Statistical3",
            "JET_EffectiveNP_Statistical4",
            "JET_EffectiveNP_Statistical5",
            "JET_EffectiveNP_Statistical6",
            "JET_JER_DataVsMC_AFII",
            "JET_JER_DataVsMC_MC16",
            "JET_JER_EffectiveNP_1",
            "JET_JER_EffectiveNP_2",
            "JET_JER_EffectiveNP_3",
            "JET_JER_EffectiveNP_4",
            "JET_JER_EffectiveNP_5",
            "JET_JER_EffectiveNP_6",
            "JET_JER_EffectiveNP_7restTerm",
            "JET_SingleParticle_HighPt",
            "JET_Pileup_OffsetMu",
            "JET_Pileup_OffsetNPV",
            "JET_Pileup_PtTerm",
            "JET_Pileup_RhoTopology",
        ]
    ],
    "MU": [
        TreeSyst("Mu_ID", "MUON_ID__1up", "MUON_ID__1down"),
        TreeSyst("Mu_MS", "MUON_MS__1up", "MUON_MS__1down"),
        TreeSyst("Mu_Sagitta_Res", "MUON_SAGITTA_RESBIAS__1up", "MUON_SAGITTA_RESBIAS__1down"),
        # TreeSyst("Mu_Sagitta_Rho", "MUON_SAGITTA_RHO__1up", "MUON_SAGITTA_RHO__1down"),
        TreeSyst("Mu_Scale", "MUON_SCALE__1up", "MUON_SCALE__1down"),
    ],
    "MET": [
        # one-sided, symmetrised (histoSysOneSideSym)
        TreeSyst("MET_Soft_reso_Para", "MET_SoftTrk_ResoPara", None),
        TreeSyst("MET_Soft_reso_Perp", "MET_SoftTrk_ResoPerp", None),
        TreeSyst("MET_Soft_Scale", "MET_SoftTrk_Scale__1up", "MET_SoftTrk_Scale__1down"),
    ],
}


def _w(name, nominal, up, down):
    return WeightSyst(name, up=up, down=down, replace=nominal)


WEIGHT_SYSTEMATICS = {
    "MU": [
        _w("muStat", "wmu_nom", "wmu_stat_up", "wmu_stat_down"),
        _w("muSys", "wmu_nom", "wmu_sys_up", "wmu_sys_down"),
        _w("muStat_lowpt", "wmu_nom", "wmu_stat_lowpt_up", "wmu_stat_lowpt_down"),
        _w("muSys_lowpt", "wmu_nom", "wmu_sys_lowpt_up", "wmu_sys_lowpt_down"),
        _w("muIsoStat", "wmu_nom", "wmu_iso_stat_up", "wmu_iso_stat_down"),
        _w("muIsoSys", "wmu_nom", "wmu_iso_sys_up", "wmu_iso_sys_down"),
        _w("muBadSys", "wmu_nom", "wmu_bad_sys_up", "wmu_bad_sys_down"),
        _w("muTTVAStat", "wmu_nom", "wmu_ttva_stat_up", "wmu_ttva_stat_down"),
        _w("muTTVASys", "wmu_nom", "wmu_ttva_sys_up", "wmu_ttva_sys_down"),
    ],
    "EG": [
        _w("elID", "wel_nom", "wel_id_up", "wel_id_down"),
        _w("elIso", "wel_nom", "wel_iso_up", "wel_iso_down"),
        _w("elReco", "wel_nom", "wel_reco_up", "wel_reco_down"),
        _w("elChFlip", "wel_nom", "wchflip_up", "wchflip_down"),
    ],
    "FT": [
        _w("FT_B", "wjet_nom", "wjet_b_up", "wjet_b_down"),
        _w("FT_C", "wjet_nom", "wjet_c_up", "wjet_c_down"),
        _w("FT_Extra1", "wjet_nom", "wjet_extra1_up", "wjet_extra1_down"),
        _w("FT_Extra2", "wjet_nom", "wjet_extra2_up", "wjet_extra2_down"),
        _w("FT_Light", "wjet_nom", "wjet_light_up", "wjet_light_down"),
        _w("JET_JVT", "wjet_nom", "wjet_jvt_up", "wjet_jvt_down"),
    ],
    "PU": [_w("pileup", "wpu_nom", "wpu_up", "wpu_down")],
    "TRIG": [
        _w("TRIG_muStat", "wmu_nom", "wmu_trig_stat_up", "wmu_trig_stat_down"),
        _w("TRIG_muSys", "wmu_nom", "wmu_trig_sys_up", "wmu_trig_sys_down"),
        _w("TRIG_elEff", "wel_nom", "wel_trig_up", "wel_trig_down"),
        _w("TRIG_elEffeff", "wel_nom", "wel_trigEff_up", "wel_trigEff_down"),
    ],
    # multiplies the nominal weight by an extra factor
    "PDF": [WeightSyst("PDF", up="wpdf_up", down="wpdf_down", replace=None)],
}


def systematic_groups():
    """Group name -> list of systematics (tree + weight)."""
    groups = {}
    for source in (TREE_SYSTEMATICS, WEIGHT_SYSTEMATICS):
        for group, systs in source.items():
            for syst in systs:
                syst.group = group
            groups.setdefault(group, []).extend(systs)
    # data-driven and theory groups are handled through the samples
    for group in ("FAKES", "CHFLIP", "TH"):
        groups.setdefault(group, [])
    return groups


# ---------------------------------------------------------------------------
# theory uncertainties (HistFitter addSRTheorySyst), values in
# share/theorySystematicsBinned.json:  thUnc_<key>Up/Down -> region -> variable
# ---------------------------------------------------------------------------

THEORY_SOURCES = [
    "muRmuF",
    "muR",
    "muF",
    "PdfAlpha",
    "QSF",
    "CKKW",
    "CSSKIN",
    "PartonShower",
    "HardScatter",
    "Radiation",
    "Scale",
    "Xsec",
]

WWSS_DSIDS = "(MCId==364287 || MCId==364286)"
NOT_WWSS_DSIDS = "(MCId!=364287 && MCId!=364286)"
FOURTOP_DSID = "(MCId==412043)"


def theory_systematics():
    """Sample name -> list of theory systematics (group TH).

    Systematics whose ``thUnc_<key>Up/Down`` entries are missing in the JSON are
    skipped automatically when the workspace is built.
    """
    th = {}
    # OtherMultiboson: sources for the WW same-sign DSIDs only + Xsec for the rest
    th["OtherMultiboson"] = [
        TheorySyst(f"theoryUncert_WWSS{src}", f"WWSS{src}", selection=WWSS_DSIDS, allow_binned=False)
        for src in THEORY_SOURCES
    ] + [
        TheorySyst(
            "theoryUncert_OtherMultibosonXsecnoWWSS",
            "OtherMultibosonXsec",
            selection=NOT_WWSS_DSIDS,
            allow_binned=False,
        )
    ]
    # SM 4 tops: only for DSID 412043
    th["SM4tops"] = [
        TheorySyst(f"theoryUncert_FourTop{src}", f"FourTop{src}", selection=FOURTOP_DSID, allow_binned=False)
        for src in THEORY_SOURCES
    ]
    # other tops: cross-section only
    th["Multitop"] = [TheorySyst("theoryUncert_MultitopXsecno4tops", "MultitopXsec", allow_binned=False)]
    # ttW: all sources + 50% flavour uncertainty on >=3b events
    th["ttW"] = [
        TheorySyst(f"theoryUncert_ttW{src}", f"ttW{src}", allow_binned=False) for src in THEORY_SOURCES
    ] + [
        WeightSyst(
            "theoryUncert_ttWFlavour",
            up="(1.5*(nBJets20>=3)+1.0*(nBJets20<3))",
            down="(0.5*(nBJets20>=3)+1.0*(nBJets20<3))",
            replace=None,
            group="TH",
        )
    ]
    # WZ: binned values available (and normalised in the CR if WZ floats)
    th["WZ"] = [
        TheorySyst(f"theoryUncert_WZ{src}", f"WZ{src}", allow_binned=True, binned_only_if_normalized=True)
        for src in THEORY_SOURCES
    ]
    # everything else: <sample><source>
    for sample in ("ttZ", "ttH", "RareTop_NottH", "Multiboson", "ttV"):
        th[sample] = [
            TheorySyst(f"theoryUncert_{sample}{src}", f"{sample}{src}", allow_binned=False)
            for src in THEORY_SOURCES
        ]
    return th


#: theory (acceptance) uncertainty of the signal, thUnc_signalUp/Down
SIGNAL_THEORY = [TheorySyst("theoryUncert_signal", "signal", allow_binned=True)]

#: RPC-meets-RPV combination: systematics removed from (full-sim) signal and the
#: extra Lxy weight systematic (HistFitter --isRPCmRPV)
RPCMRPV_SIGNAL_EXCLUDE = ["JET_RelativeNonClosure_AFII", "JET_JER_DataVsMC_AFII"]
RPCMRPV_SIGNAL_EXTRA = [
    WeightSyst(
        "LxyFactor",
        up="wSystFactorLxyWeight_up",
        down="wSystFactorLxyWeight_down",
        replace=None,
        group="SIG",
    )
]
