"""SS/3L analysis definitions (port of the HistFitter ``fitConfigSS3L.py``).

``build_analysis(options)`` is the entry point used by the workspace builder.
Edit ``regions.py`` (cuts, binnings), ``samples.py`` (samples, weights, data-driven
yields), ``systematics.py`` (systematic catalogue) and ``signals.py`` (grids) to
change the physics content; the YAML configuration only selects what enters a
workspace.

``options`` comes from ``General.Options`` of the workspace YAML, e.g.
``{"isRPCmRPV": true}``.
"""

from fit2l3l.analysis.base import Analysis
from fit2l3l.analysis.ss3l import regions, samples, signals, systematics
from fit2l3l.analysis.ss3l.plotting import PLOTTING_MODELS
from fit2l3l.utils import read_json, share_path

#: hard-coded yields (data-driven backgrounds, data counts, NFs)
HARDCODED_YIELDS = "hardCodedYields.json"
#: theory uncertainties (relative, per region and bin)
THEORY_VALUES = "theorySystematicsBinned.json"


def build_analysis(options=None) -> Analysis:
    options = dict(options or {})
    hardcoded = read_json(share_path(options.get("HardCodedYields", HARDCODED_YIELDS)))
    theory_values = read_json(share_path(options.get("TheoryValues", THEORY_VALUES)))
    nfs = {k: v for k, v in hardcoded.get("NF", {}).items() if not k.startswith("!")}

    def _grid(grid, schema="SP:4,LSP:5", exclude=()):
        return signals.read_grid(grid, schema, exclude)

    def _signal_sample(point, grid):
        return signals.signal_sample(point, grid, options)

    return Analysis(
        name="ss3l",
        regions=regions.build_regions(),
        samples=samples.build_samples(),
        nominal_weights=list(systematics.NOMINAL_WEIGHTS),
        systematic_groups=systematics.systematic_groups(),
        all_groups=list(systematics.ALL_GROUPS),
        theory=systematics.theory_systematics(),
        theory_values=theory_values,
        hardcoded_yields=hardcoded,
        normalization_factors=nfs,
        signal_grid=_grid,
        signal_sample=_signal_sample,
        histogram_path=samples.histogram_path,
        data_sample="data",
        plotting_models=PLOTTING_MODELS,
        options=options,
    )
