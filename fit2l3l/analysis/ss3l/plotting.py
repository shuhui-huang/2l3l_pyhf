"""Plotting configuration of the signal models (port of ``plottingConfig.py``).

Labels may use ROOT TLatex syntax (``#tilde{g}``) - they are converted to
matplotlib mathtext automatically - or matplotlib mathtext directly.

Keys of a model:

* ``x_range`` / ``y_range``         : axis ranges of the contour plot
* ``contour_range``                 : (xmin, xmax, ymin, ymax) used for the
  interpolation (HistFitter ``setRangeForContours``), optional
* ``labels``                        : (x label, y label, process description)
* ``forbidden``                     : kinematically forbidden region, line
  ``y = x - cut`` (or a custom ``function`` of x given as a python expression),
  with a label at (``label_x``, ``label_y``)
* ``previous_limits``               : list of {label, csv, line, scale} (HEPData csv
  with two columns x,y; ``line=True`` for 1D cross-section plots, ``scale`` converts
  the units)
* ``extra_text`` / ``theory_legend``: additional legend text for 1D plots
* ``log_y``                         : log scale (1D cross-section limits)
"""

PREVIOUS_36 = "#splitline{SS/3L obs. 36 fb^{-1}}{[arXiv:1706.03731]}"
PREVIOUS_20 = "#splitline{SS/3L obs. 20.3 fb^{-1}}{[arXiv:1404.2500]}"
PREVIOUS_139 = "#splitline{SS/3L obs. 139 fb^{-1}}{[arXiv:1909.08457]}"

PLOTTING_MODELS = {
    "Mc16SusyBtt": dict(
        x_range=(600, 1100),
        y_range=(50, 1100),
        contour_range=(500, 1100, 30, 1100),
        labels=(
            "m(#tilde{b}_{1}) [GeV]",
            "m(#tilde{#chi}^{0}_{1}) [GeV]",
            "#tilde{b}_{1} #tilde{b}_{1} production, #tilde{b}_{1}#rightarrow t#tilde{#chi}^{#pm}_{1}, "
            "m(#tilde{#chi}^{#pm}_{1}) = m(#tilde{#chi}^{0}_{1}) + 100 GeV",
        ),
        forbidden=dict(cut=272.5, label_x=700, label_y=480,
                       text="m(#tilde{b}_{1}) < m(t) + m(#tilde{#chi}^{0}_{1}) + 100 GeV"),
        previous_limits=[dict(label=PREVIOUS_36, csv="HEPData/Mc16SusyBtt_36ifb.csv")],
    ),
    "Mc16SusyGtt": dict(
        x_range=(1100, 2200),
        y_range=(0, 2200),
        labels=(
            "m_{#tilde{g}} [GeV]",
            "m_{#tilde{#chi}^{0}_{1}} [GeV]",
            "#tilde{g} #tilde{g} production, #tilde{g}#rightarrow t#bar{t}#tilde{#chi}^{0}_{1}, m(#tilde{t}_{1}) >> m(#tilde{g})",
        ),
        forbidden=dict(cut=345, label_x=1400, label_y=1100, text="m_{#tilde{g}} < 2 m_{t} + m_{#tilde{#chi}^{0}_{1}}"),
    ),
    "Mc16SusyComprGtt": dict(
        x_range=(750, 2000),
        y_range=(5, 150),
        labels=(
            "m_{#tilde{g}} [GeV]",
            "#Deltam = m_{#tilde{#chi}^{0}_{1}} + 2m_{t} - m_{#tilde{g}} [GeV]",
            "#tilde{g} #tilde{g} production, #tilde{g}#rightarrow t#bar{t}#tilde{#chi}^{0}_{1}",
        ),
        forbidden=dict(cut=160, label_x=810, label_y=690, text="m_{#tilde{g}} < 2 m_{W} + m_{#tilde{#chi}^{0}_{1}}"),
    ),
    "Mc16SusyGSL": dict(
        x_range=(600, 2200),
        y_range=(200, 2200),
        labels=(
            "m_{#tilde{g}} [GeV]",
            "m_{#tilde{#chi}^{0}_{1}} [GeV]",
            "#tilde{g} #tilde{g} production, #tilde{g} #rightarrow qq(ll/#nu#nu)#tilde{#chi}^{0}_{1}",
        ),
        forbidden=dict(cut=0, label_x=800, label_y=850, text="m(#tilde{g}) < m(#tilde{#chi}^{0}_{1})"),
    ),
    "Mc16SusyGG2StepWZ": dict(
        x_range=(1000, 2300),
        y_range=(100, 2600),
        labels=(
            "m(#tilde{g}) [GeV]",
            "m(#tilde{#chi}^{0}_{1}) [GeV]",
            "#tilde{g} #tilde{g} production, #tilde{g} #rightarrow qq'WZ#tilde{#chi}^{0}_{1}; "
            "m(#tilde{#chi}^{#pm}_{1}) = (m(#tilde{g}) + m(#tilde{#chi}^{0}_{1}))/2, "
            "m(#tilde{#chi}^{0}_{2}) = (m(#tilde{#chi}^{#pm}_{1}) + m(#tilde{#chi}^{0}_{1}))/2",
        ),
        forbidden=dict(cut=320, label_x=1400, label_y=1150, function="x",
                       text="#Deltam(#tilde{#chi}_{1}^{#pm},#tilde{#chi}_{2}^{0}) < m(W), "
                            "#Deltam(#tilde{#chi}_{2}^{0},#tilde{#chi}_{1}^{0}) < m(Z)"),
        previous_limits=[dict(label=PREVIOUS_36, csv="HEPData/Mc16SusyGG2StepWZ_36ifb.csv")],
    ),
    "Mc16SusyGGN2_SLN1": dict(
        x_range=(600, 2600),
        y_range=(100, 3100),
        labels=(
            "m(#tilde{g}) [GeV]",
            "m(#tilde{#chi}^{0}_{1}) [GeV]",
            "#tilde{g} #tilde{g} production, #tilde{g} #rightarrow q#bar{q}(ll/#nu#nu)#tilde{#chi}^{0}_{1}",
        ),
        forbidden=dict(cut=320, label_x=1300, label_y=1320, function="x", text="m(#tilde{g}) < m(#tilde{#chi}_{1}^{0})"),
        previous_limits=[dict(label="#splitline{SS/3L obs. 36.1 fb^{-1}}{[JHEP09(2017)084]}", csv="HEPData/GGN2SLN136_1.csv")],
    ),
    "Mc16SusyGG_Rpv331": dict(
        x_range=(600, 2300),
        y_range=(400, 2100),
        labels=(
            "m(#tilde{g}) [GeV]",
            "m(#tilde{t}) [GeV]",
            "#tilde{g} #tilde{g} production, #tilde{g} #rightarrow #tilde{t} #bar{t}, #tilde{t} #rightarrow #bar{b} #bar{d}",
        ),
        forbidden=dict(cut=172.5, label_x=1000, label_y=870, text="m(#tilde{g}) < m(#tilde{t}) + m(t)"),
        previous_limits=[dict(label=PREVIOUS_139, csv="HEPData/Mc16Susy_Rpv331_139ifb.csv")],
    ),
    "Mc16SusyGG_Rpv321": dict(
        x_range=(800, 1600),
        y_range=(400, 1600),
        labels=(
            "m_{#tilde{g}} [GeV]",
            "m_{#tilde{t}} [GeV]",
            "#tilde{g} #tilde{g} production, #tilde{g} #rightarrow #tilde{t} #bar{t}, #tilde{t} #rightarrow #bar{d} #bar{s}",
        ),
        forbidden=dict(cut=0, label_x=700, label_y=640, text="m_{#tilde{d}} = m_{#tilde{g}}"),
    ),
    "Mc16SusyTT2Step": dict(
        x_range=(550, 950),
        y_range=(8e-3, 4),
        log_y=True,
        labels=(
            "m(#tilde{t}_{1}) [GeV]",
            "#sigma [pb]",
            "#tilde{t}_{1}#tilde{t}_{1} production, #tilde{t}_{1} #rightarrow tW^{#pm}#tilde{#chi}^{#pm}_{1}, "
            "#tilde{#chi}^{#pm}_{1} #rightarrow W*#tilde{#chi}^{0}_{1}",
        ),
        previous_limits=[dict(label=PREVIOUS_36, csv="HEPData/Mc16SusyTT2Step_36ifb.csv", line=True, scale=0.001)],
        extra_text="BR(#tilde{t}_{1}#rightarrow tW^{#pm}(W*)#tilde{#chi}_{1}^{0})=100%",
        theory_legend="pp #rightarrow #tilde{t}_{1} #tilde{t}_{1}",
    ),
    "Mc16SusyGG_RpvLQD": dict(
        x_range=(1000, 3000),
        y_range=(50, 3300),
        labels=(
            "m(#tilde{g}) [GeV]",
            "m(#tilde{#chi}^{0}_{1}) [GeV]",
            "#tilde{g} #tilde{g} production, #tilde{g} #rightarrow qq#tilde{#chi}^{0}_{1}, #tilde{#chi}^{0}_{1} #rightarrow lqq",
        ),
        forbidden=dict(cut=0, label_x=1300, label_y=1400, text="m(#tilde{g}) < m(#tilde{#chi}^{0}_{1})"),
        previous_limits=[dict(label=PREVIOUS_36, csv="HEPData/Mc16SusyGG_RpvLQD_36ifb.csv")],
    ),
    "Mc16SusyC1N2_onshellWZ": dict(
        x_range=(100, 600),
        y_range=(0, 600),
        contour_range=(100, 400, 0, 600),
        labels=(
            "m(#tilde{#chi}^{#pm}_{1}/#tilde{#chi}^{0}_{2}) [GeV]",
            "m(#tilde{#chi}^{0}_{1}) [GeV]",
            "#tilde{#chi}^{#pm}_{1}#tilde{#chi}^{0}_{2} #rightarrow #tilde{#chi}^{0}_{1}#tilde{#chi}^{0}_{1}WZ (on-shell)",
        ),
        forbidden=dict(cut=91, label_x=250, label_y=180, text="#Deltam(#tilde{#chi}^{0}_{2},#tilde{#chi}^{0}_{1}) < m(Z)"),
    ),
    "Mc16SusyC1N2_offshellWZ": dict(
        x_range=(100, 600),
        y_range=(0, 600),
        contour_range=(100, 100, 0, 350),
        labels=(
            "m(#tilde{#chi}^{#pm}_{1}/#tilde{#chi}^{0}_{2}) [GeV]",
            "m(#tilde{#chi}^{0}_{1}) [GeV]",
            "#tilde{#chi}^{#pm}_{1}#tilde{#chi}^{0}_{2} #rightarrow #tilde{#chi}^{0}_{1}#tilde{#chi}^{0}_{1}WZ",
        ),
        forbidden=dict(cut=191, label_x=250, label_y=180, text="#Deltam(#tilde{#chi}^{0}_{2},#tilde{#chi}^{0}_{1}) < m(Z)"),
    ),
    "Mc16SusyC1N2_allWZ": dict(
        x_range=(100, 600),
        y_range=(0, 600),
        contour_range=(100, 600, 0, 600),
        labels=(
            "m(#tilde{#chi}^{#pm}_{1}/#tilde{#chi}^{0}_{2}) [GeV]",
            "m(#tilde{#chi}^{0}_{1}) [GeV]",
            "#tilde{#chi}^{#pm}_{1}#tilde{#chi}^{0}_{2} #rightarrow #tilde{#chi}^{0}_{1}#tilde{#chi}^{0}_{1}WZ",
        ),
        forbidden=dict(cut=0, label_x=200, label_y=200, function="x", text="#Deltam(#tilde{#chi}^{#pm}_{1},#tilde{#chi}^{0}_{1}) < 0"),
    ),
    "Mc16SusyC1N2N1_GGMHinoZh": dict(
        x_range=(200, 900),
        y_range=(0, 100),
        labels=(
            "m(#tilde{#chi}^{#pm}_{1}/#tilde{#chi}^{0}_{2}/#tilde{#chi}^{0}_{1}) [GeV]",
            "B(#tilde{#chi}^{0}_{1} #rightarrow Z #tilde{G}) [%]",
            "Higgsino GGM, #tilde{#chi}^{0}_{1} #rightarrow Z/h#tilde{G}",
        ),
    ),
    "Mc16SusySS2StepWZ": dict(
        x_range=(500, 1800),
        y_range=(100, 2200),
        labels=(
            "m(#tilde{q}) [GeV]",
            "m(#tilde{#chi}^{0}_{1}) [GeV]",
            "#tilde{q} #tilde{q} production, #tilde{q} #rightarrow qWZ#tilde{#chi}^{0}_{1}",
        ),
        forbidden=dict(cut=320, label_x=1225, label_y=975, function="x",
                       text="#Deltam(#tilde{#chi}_{2}^{0},#tilde{#chi}_{1}^{0}) < m(Z)"),
        previous_limits=[dict(label=PREVIOUS_20, csv="HEPData/Mc16SusySS2StepWZ_20ifb.csv")],
    ),
    "Mc16SusySS2StepSL": dict(
        x_range=(600, 2000),
        y_range=(100, 2200),
        labels=(
            "m(#tilde{q}) [GeV]",
            "m(#tilde{#chi}^{0}_{1}) [GeV]",
            "#tilde{q} #tilde{q} production, #tilde{q} #rightarrow q(l#nu/ll)#tilde{#chi}^{0}_{1}",
        ),
        forbidden=dict(cut=0, label_x=1000, label_y=1100, text="m(#tilde{q}) < m(#tilde{#chi}_{1}^{0})"),
        previous_limits=[dict(label=PREVIOUS_20, csv="HEPData/Mc16SusySS2StepSL_20ifb.csv")],
    ),
    "Mc16SusyGTTLLrpvHF": dict(
        x_range=(-2, 2.2),
        y_range=(1100, 3000),
        labels=("log(#tau [ns])", "m(#tilde{g}) [GeV]", "GTTLL"),
    ),
    "Mc16SusyTTLLrpvHF": dict(
        x_range=(-2, 2.2),
        y_range=(400, 2200),
        labels=("log(#tau [ns])", "m(#tilde{t}) [GeV]", "TTLL"),
    ),
    "Mc16SusyGTTrpvHF": dict(
        x_range=(1400, 2400),
        y_range=(8e-5, 1),
        log_y=True,
        labels=(
            "m(#tilde{g}) [GeV]",
            "#sigma [pb]",
            "#tilde{g}#tilde{g} production, #tilde{g} #rightarrow tt#tilde{#chi}^{0}_{1}, #tilde{#chi}^{0}_{1} #rightarrow tbs",
        ),
        theory_legend="pp #rightarrow #tilde{g} #tilde{g}",
    ),
    "Mc16SusyTTrpvHF": dict(
        x_range=(400, 1800),
        y_range=(8e-5, 500),
        log_y=True,
        labels=(
            "m(#tilde{t}) [GeV]",
            "#sigma [pb]",
            "#tilde{t}#tilde{t} production, #tilde{t} #rightarrow t#tilde{#chi}^{0}_{1}, #tilde{#chi}^{0}_{1} #rightarrow tbs",
        ),
        theory_legend="pp #rightarrow #tilde{t} #tilde{t}",
    ),
}
