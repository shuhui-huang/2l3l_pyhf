"""fit2l3l: statistical analysis framework for the 2L/3L (SS/3L) analyses.

Workflow::

    ntuples / histograms --(cabinetry, python analysis definitions)--> ws.json
    ws.json              --(pyhf, fit YAML)--> fits, limits, tables, plots

* :mod:`fit2l3l.analysis`  - physics definitions (regions, samples, systematics)
* :mod:`fit2l3l.workspace` - cabinetry workspace building
* :mod:`fit2l3l.fitting`   - pyhf fits steered by YAML
* :mod:`fit2l3l.plotting`  - contours, upper limits, summary plots
"""

__version__ = "0.1.0"
