# 2l3l_pyhf

Statistical interpretation for the 2L/3L (SS/3L) analyses with
[cabinetry](https://github.com/scikit-hep/cabinetry) + [pyhf](https://github.com/scikit-hep/pyhf).

`fit2l3l` replaces the HistFitter wrapper `ss3l_hf` (`generateCommands.py` +
`fitConfigSS3L.py` + HistFitter scripts) with a pure-python workflow:

```
 ntuples / histograms ──(cabinetry + python analysis definitions)──▶ ws.json (+ patchset.json)
                                                                        │
                                       fit YAML ──(pyhf)──▶ fits, CLs, limits, tables, plots, contours
```

* **Workspace YAML** (cabinetry step) – *what* goes into the workspace: the regions
  with their role (SR/CR/VR), the samples, systematic groups, blinding, NFs, …
* **Python analysis definitions** – *how*: region cuts and binnings, sample trees and
  weights, systematic definitions, data-driven yields, signal grids
  (`fit2l3l/analysis/ss3l/`, the port of `fitConfigSS3L.py`).
* **Fit YAML** (pyhf step) – which fits run on `ws.json`: background-only, discovery,
  exclusion (+ contours), options for toys, limits, blinding, tables and plots.

Contents: [Setup](#setup) · [Quick start](#quick-start) · [Workspace YAML](#workspace-yaml-cabinetry) ·
[Python definitions](#python-analysis-definitions) · [Fit YAML](#fit-yaml-pyhf) · [Outputs](#outputs) ·
[Commands](#commands) · [HistFitter → fit2l3l](#histfitter--fit2l3l) · [Differences to HistFitter](#differences-to-histfitter-read-before-validating) ·
[Development](#development)

---

## Setup

```bash
git clone -b pyhf-dev https://github.com/shuhui-huang/2l3l_pyhf.git
cd 2l3l_pyhf
source setup.sh          # first time: creates .venv and installs fit2l3l + cabinetry + pyhf
```

Any python ≥ 3.9 works (lxplus EL9 `python3` is fine, no ROOT needed). Use
`FIT2L3L_VENV=/path/to/venv source setup.sh` to keep the environment outside the
repository. Without the script: `pip install -e .`.

## Quick start

```bash
# toy ntuples in the HistFitter input format, then the full chain (a few minutes)
bash examples/run_toy_example.sh
```

With real inputs (HistFitter-format ntuples `<sample>.<lumi>.root` in `InputTrees/`):

```bash
# 1) cabinetry: build the workspace -> results/bkgonly_RpcSSwz/ws.json
fit2l3l build configs/workspace/bkgonly_RpcSSwz.yaml
fit2l3l inspect results/bkgonly_RpcSSwz          # regions, samples, yields

# 2) pyhf: run the fits defined in a fit YAML on ws.json
fit2l3l fit configs/fits/bkgonly.yaml

# discovery fit / model-independent limits
fit2l3l build configs/workspace/disc_RpcGGwz2.yaml
fit2l3l fit   configs/fits/discovery.yaml

# exclusion over a signal grid: ws.json (bkg-only) + patchset.json (one patch per point)
fit2l3l build configs/workspace/excl_GG2StepWZ_RpcGGwz.yaml -j 4
fit2l3l fit   configs/fits/exclusion.yaml -j 4         # CLs, limits, merge, contour plot
```

Any `General` option of a workspace YAML can be overridden on the command line,
e.g. `--set Systematics=NONE --set Blind=[SR,VR] --set InputPath=/eos/...`.

---

## Workspace YAML (cabinetry)

`configs/workspace/*.yaml`. Only selections and switches; everything else is in python.

```yaml
General:
  Name: bkgonly_RpcSSwz       # output: <OutputDir>/<Name>/ws.json
  Analysis: ss3l              # python definitions: fit2l3l/analysis/ss3l (or a dotted module path)
  FitType: bkg                # bkg | excl | disc
  Lumi: 138965                # pb^-1; input files are <sample>.<Lumi>.root
  LumiUncertainty: 0.017
  InputPath: InputTrees       # directory with the ntuples (InputType: histogram -> ROOT file)
  InputType: ntuple           # ntuple | histogram
  OutputDir: results
  Systematics: ALL            # ALL | NONE | EG,JET,MU,MET,FT,PU,TRIG,PDF,FAKES,CHFLIP,TH,DDSTAT
                              # | single systematic names | FLAT30 | DDFLAT30
  Blind: [SR]                 # SR, CR, VR, ALL: data replaced by the background expectation
  StatConstraint: Gaussian    # Gaussian | Poisson (stat. uncertainty of data-driven backgrounds)
  BinnedTheoryUncertainty: false
  POI: mu_SIG
  POIBounds: [0, 10]          # default [0, 10] (excl), [0, 100] (disc)
  NormalizeSystematicsInCR: true   # HistFitter "Norm" systematics for free-floating backgrounds
  MissingTree: nominal        # nominal | error : missing systematic trees/histograms
  MissingYields: error        # error | zero : region missing in the hard-coded (data-driven) yields
  Options: {isRPCmRPV: false} # passed to the analysis module

Regions:
  CR: [CRWZ2j]
  VR: [VRWZ4j, VRWZ6j]
  SR:
    - {Name: RpcSSwz1, Variable: met}     # binned fit (binning from python), optional Overflow/Underflow
    - RpcSSwz4                            # single bin ("cuts")

Samples:
  Data: ntuple                # ntuple | hardcoded (dataCounts of share/hardCodedYields.json)
  Backgrounds: [Multitop, SM4tops, OtherMultiboson, ttW, ttZ, WZ, MisCharge, Fakes]
  Signal:                     # FitType: excl only
    Grid: Mc16SusyGG2StepWZ   # share/susyGridFiles/<Grid>
    SigNameSchema: "SP:4,LSP:7"
    Points: all               # or [376776, ...]
    Exclude: []

NormFactors:                  # free normalisation in CR(s)    (HF --bkgToNormalize WZ:CRWZ2j)
  WZ: CRWZ2j                  # long form: [{Sample: WZ, Regions: [CRWZ2j], Name: mu_WZ, Bounds: [0, 5]}]
FixedNF: [WZ]                 # scale by the NF of hardCodedYields.json + its uncertainty (HF --NF)
ExtraSampleWeights:           # (HF --extraSampleWeight)
  WZ: weightSpecial
Pruning: {Norm: 0.0, Shape: 0.0}   # drop normsys/histosys with smaller relative effect
```

**Region roles** follow the HistFitter conventions:

| regions given | bkg-only fit | exclusion / discovery |
|---|---|---|
| SR | SR fitted (as CR) | SR fitted |
| VR | VR fitted (as CR) | – |
| CR (+SR, +VR) | CR fitted, SR/VR extrapolated | CR + SR fitted, VR only post-fit |

Under-/overflow are added to the first/last bin for CRs and VRs, and to the last bin of
binned SRs (HistFitter `useOverflowBin`/`useUnderflowBin`).

**What `fit2l3l build` does**
1. composes a full cabinetry configuration from the YAML + python definitions
   (written to `cabinetry_config.yaml` for inspection);
2. `cabinetry.templates.build` with a cabinetry `Router` whose template builder reads each
   (file, tree) once and fills all histograms from it (C++ cut strings understood,
   under/overflow, data-driven yields from JSON, theory uncertainties per region/bin);
   `cabinetry.templates.postprocess`; re-normalisation of "Norm" systematics in the CRs;
3. `cabinetry.workspace.build`, then: `lumi` modifier (`setNormByTheory`), Poisson stat.
   uncertainties, pruning/sanitising, blinding; validation with pyhf;
4. exclusion: background-only `ws.json` + `patchset.json` with one pyhf patch per signal
   point (HEPData `BkgOnly.json` + `patchset.json` format; `pyhf patchset apply` works).

## Python analysis definitions

`fit2l3l/analysis/ss3l/` (port of `fitConfigSS3L.py`, `hardCodedYields.json`, `plottingConfig.py`):

| file | content | HistFitter equivalent |
|---|---|---|
| `regions.py` | `CUTS` (C++ selections, copied verbatim) and `BINNINGS` (`(nbins, low, high)` or edges) | `configMgr.cutsDict`, `binning` in `hardCodedYields.json` |
| `samples.py` | samples: file/tree pattern, colour, label, weights, data-driven yield keys, sample-specific systematics; `histogram_path()` for histogram inputs | `sample(...)`, `allBkg.addSample`, `Helper.py` |
| `systematics.py` | nominal weights, tree systematics (`TreeSyst`), weight systematics (`WeightSyst`), theory systematics per sample (`TheorySyst`), groups, RPCmRPV options | `GetTreeSystematics`, `GetWeightSystematics`, `addSRTheorySyst` |
| `signals.py` | grid files, mass parsing (`SigNameSchema`), cross-sections (`share/xsections`), grid-specific weights | `signalSameSignTools.py`, `FillJsonFilesWithMissingInfo.py` |
| `plotting.py` | contour plot settings per model (ranges, labels, kinematic limit, previous limits) | `plottingConfig.py` |
| `__init__.py` | `build_analysis(options)` entry point, JSON inputs | – |

Inputs in `share/`: `hardCodedYields.json` (data-driven yields, data counts, NFs),
`theorySystematicsBinned.json`, `susyGridFiles/`, `xsections/`, `HEPData/`.

Adding things:
* **region**: add the cut to `CUTS` (+ binning to `BINNINGS`), list it in a workspace YAML;
* **sample**: add a `Sample(...)` in `samples.py`, list it in `Samples.Backgrounds`;
* **systematic**: add a `TreeSyst`/`WeightSyst` to a group in `systematics.py`
  (or a new group), switch it on with `General.Systematics`;
* **another analysis** (e.g. a new 2L3L analysis): copy `analysis/ss3l` to
  `analysis/<name>`, edit, and set `General.Analysis: <name>`.

Conventions of the generated cabinetry config: `Tree: "{prefix}_X"` (systematic tree
of the sample), `Weight: "{weight}*f"` / `"{weight:a->b}"` (nominal weight times `f` /
with factor `a` replaced by `b`); systematics with empty `Up`/`Down` are built by
fit2l3l (data-driven yields, JSON theory uncertainties).

## Fit YAML (pyhf)

`configs/fits/*.yaml`. Several fits can be listed under `Fits`.

```yaml
General:
  Workspace: results/bkgonly_RpcSSwz   # directory with ws.json + meta.json (or --workspace)
  OutputDir: null                      # default: <Workspace>/fits/<fit Name>
  Optimizer: scipy                     # hypothesis tests (scipy is faster); fits use minuit
  InterpolationCodes: {}               # pyhf defaults (normsys code4, histosys code4p); e.g. {normsys: code1, histosys: code0}
  ATLASLabel: Internal
  PlotFormats: [pdf, png]

Fits:
  - Name: bkgonly
    Type: bkg                 # bkg | disc | excl (default: type of the workspace)
    FitRegions: null          # default from the region roles; roles (CR/SR/VR) or names
    ValidationRegions: null
    Blind: []                 # fit-time blinding (roles or names)
    Asimov: false             # fit the background-only Asimov dataset
    Fixed: {}                 # {parameter: value};  Init: {}, Bounds: {par: [lo, hi]}
    Minos: [mu_WZ]
    Plots: [prefit, postfit, summary, pulls, correlation, modifier_grid]
    Tables: [yields, systematics]
    Ranking: {Enabled: false, POI: mu_WZ, MaxPars: 20}
    # discovery / exclusion
    Calculator: asymptotics   # asymptotics | toybased
    TestStat: qtilde
    NToys: 1000
    Seed: 0
    SaveToys: false           # store toy distributions, merge jobs with `fit2l3l merge`
    CL: 0.95
    UpperLimit: {Enabled: true, Scan: null, NPoints: 11}   # Scan: null (auto range) | [lo, hi, n] | toms748
    TheoryVariations: true    # excl: observed CLs for signal xsec +-1 sigma
    Points: all               # excl: signal points
    Jobs: 1                   # excl: parallel processes

Contours:                     # exclusion only (also: fit2l3l contours)
  Model: Mc16SusyGG2StepWZ
  Interpolation: linear       # linear | cubic | rbf
  DrawCLs: false
  DrawXS: false
  OneD: false                 # 1D upper limit on the cross-section
  BestSR: {RpcGGwz1: results/excl_GG2StepWZ_RpcGGwz1/fits/exclusion/hypotest_list.json}
```

## Outputs

`results/<Name>/` (workspace): `ws.json`, `patchset.json` (excl), `meta.json`,
`cabinetry_config.yaml`, `workspace_config.yaml`, `histograms/`.

`results/<Name>/fits/<fit>/`:

| fit | outputs |
|---|---|
| bkg | `fit_results.json/.txt`, `summary.json`, `plots/<region>_{prefit,postfit}`, `plots/summary_*`, `pulls`, `correlation_matrix`, `ranking`, `uncertainties_grouped_postfit`, `tables/yields_{fit,validation}_regions.{txt,tex,json}`, `tables/systematics_{prefit,postfit}.{txt,tex,json}` |
| disc | the above + `discovery.json` (p0, Z, S95 obs/exp, visible xsec, CLb) and `upper_limit_table.{txt,tex}` |
| excl | `points/<point>.json`, `hypotest_list.json` (HistFitter names: `CLs`, `CLsexp`, `clsd1s`, `clsu1s`, …, `upperLimit`, `expectedUpperLimit*`, `excludedXsec`, `m0`, `m12`, `fID`, `xsec`, `signal_yields`), `missing_points.txt`, `plots/<grid>_contour.{pdf,png}` + `_contours.json`, toys in `toys/` |

## Commands

| command | purpose |
|---|---|
| `fit2l3l build <ws.yaml> [--set K=V] [--points ...] [-j N] [--full-workspaces]` | build `ws.json` (+ `patchset.json`) |
| `fit2l3l inspect <ws dir>` | regions, roles, yields, modifiers |
| `fit2l3l fit <fit.yaml> [--workspace D] [--fit NAME] [--points ...] [--toys N --seed S] [-j N] [--set K=V]` | run fits |
| `fit2l3l run <ws.yaml> <fit.yaml>` | build + fit |
| `fit2l3l merge <fit dir> [--workspace D]` | merge exclusion points and toy files, list missing points; for a discovery fit dir: merge the upper-limit toys |
| `fit2l3l contours <hypotest_list.json> --workspace D [--best SR=list.json] [--draw-cls] [--draw-xs] [--1d]` | contours, best-SR, 1D limits |
| `fit2l3l batch <fit.yaml> [--points-per-job N] [--seeds 1-20 --toys N] [--system condor\|slurm\|local] [--queue workday] [--points-file missing_points.txt] [--submit]` | batch jobs |
| `fit2l3l toy-inputs [...]` | toy ntuples for testing |

Batch example (lxplus):
```bash
fit2l3l batch configs/fits/exclusion.yaml --points-per-job 5 --queue workday --submit
fit2l3l merge results/excl_GG2StepWZ_RpcGGwz2/fits/exclusion --workspace results/excl_GG2StepWZ_RpcGGwz2
fit2l3l batch configs/fits/exclusion.yaml --points-file results/.../missing_points.txt --submit   # resubmit
# toys split over seeds (SaveToys: true, Calculator: toybased)
fit2l3l batch configs/fits/exclusion.yaml --seeds 1-20 --toys 500 --submit
```

---

## HistFitter → fit2l3l

| HistFitter wrapper (`generateCommands.py` / `fitConfigSS3L.py`) | fit2l3l |
|---|---|
| `--fit --fitmodel bkg/excl/disc` | `General.FitType` + `fit2l3l build`, then `fit2l3l fit` |
| `--SR/--CR/--VR` | `Regions: {SR: [...], CR: [...], VR: [...]}` |
| `--Sys ALL/None/EG,JET,…/FLAT30/DDFLAT30` | `General.Systematics` (same tokens) |
| `--doBlind SR,CR,VR,ALL` | `General.Blind` (build time) or fit `Blind` |
| `--binnedFitSR/CR/VR R:var` | `{Name: R, Variable: var}` |
| `--bkgToNormalize WZ:CRWZ2j` | `NormFactors: {WZ: CRWZ2j}` |
| `--NF WZ` | `FixedNF: [WZ]` |
| `--extraSampleWeight WZ:w` | `ExtraSampleWeights: {WZ: w}` |
| `--StatConstraintType Poisson` | `General.StatConstraint: Poisson` |
| `--BinnedTheoryUncertainty` | `General.BinnedTheoryUncertainty: true` |
| `--isRPCmRPV` | `General.Options: {isRPCmRPV: true}` |
| `--Model`, `--SigNameSchema`, `--signalList`, `excludedPoints` | `Samples.Signal: {Grid, SigNameSchema, Points, Exclude}` |
| `useDataSample = False` (hard-coded data) | `Samples.Data: hardcoded` |
| `lumiTag`, `lumiErr` | `General.Lumi`, `General.LumiUncertainty` |
| `calculatorType 0/2`, `--numTOYs`, `--ToySeed` | fit `Calculator`, `NToys`, `Seed`, `SaveToys` |
| `testStatType 3`, `nPoints`, `--muScanRange` | fit `TestStat: qtilde`, `UpperLimit: {NPoints, Scan}`, `POIBounds` |
| `--doUL` | fit `UpperLimit.Enabled` |
| theory `Nominal/Up/Down` runs | fit `TheoryVariations` (mu = 1 ± Δσ/σ, one fit per point) |
| `--merge` | automatic, or `fit2l3l merge` |
| `--plot`, `--drawCLs`, `--drawXS`, `--makeBest`, `--draw1D` | `Contours:` block / `fit2l3l contours --draw-cls --draw-xs --best --1d` |
| `--batch --extra syst:condor,queue:X`, `--resubmit`, `CheckJobs.C` | `fit2l3l batch --system condor --queue X`, `--points-file missing_points.txt` |
| `--yields` (YieldsTable), `--SysBreak` (SysTable), `--FitResults` | fit `Tables: [yields, systematics]`, `fit_results.json`, `Ranking` |
| `-D before,after,corrMatrix`, `pull_maker.py` | fit `Plots: [prefit, postfit, correlation, pulls]` |
| `plotSR.py`, `PlotSystematicsGrouped.py` | `plots/summary_*`, `plots/uncertainties_grouped_*` |
| `UpperLimitTable.py` | discovery fit: `upper_limit_table.{txt,tex}` |
| `toys_helper/` | `SaveToys` + `fit2l3l batch --seeds` + `fit2l3l merge` |
| `--debug DSID` | `fit2l3l fit ... --points DSID -vv` |
| HistFitter cache / `--UseCacheHisto` | `results/<Name>/histograms/` (cabinetry templates) |

## Differences to HistFitter (read before validating)

* **Systematic decomposition.** Shape systematics are cabinetry `NormPlusShape`:
  a `normsys` (normalisation) + `histosys` (shape at fixed normalisation) sharing one
  parameter, instead of a single HistFactory `HistoSys`. pyhf's default interpolation is
  `code4` (normsys) / `code4p` (histosys); use `General.InterpolationCodes` to change.
* **Nominal MC weight.** In `fitConfigSS3L.py` the check
  `if 'TTbarSgTop' or 'VJets' in name:` is always true, so *every* MC background was
  weighted with `totweight*lumiScaling` instead of the product of the weight branches.
  fit2l3l uses `NOMINAL_WEIGHTS` (product) for all MC except TTbarSgTop/VJets. If
  `totweight` is not exactly that product in the ntuples, yields will differ.
* **Signal cross-section uncertainty band** is computed by testing mu = 1 ± Δσ/σ instead
  of re-running the fit with `lumiScaling_up/down` weights (equivalent for a
  normalisation-only variation).
* **"Norm" systematics** (re-normalised in the CR) are applied to all free-floating
  backgrounds (`NormalizeSystematicsInCR`), HistFitter did this for WZ only.
* **Flat systematics** (`FLAT<n>`) act on the backgrounds only; HistFitter also applied
  them to the signal.
* **Data-driven stat. uncertainties** are switched on together with their systematics
  group (FAKES/CHFLIP) as in HistFitter; add `DDSTAT` to get them alone.
  `StatConstraint: Poisson` uses pyhf `shapesys` for them; the MC statistical
  uncertainty stays Gaussian (`staterror`; pyhf has no Poisson Barlow-Beeston-lite).
* **Missing inputs**: a theory uncertainty without a value for a region has no effect
  (warning), a missing systematic tree uses the nominal (warning; `MissingTree: error`
  to fail). Missing data-driven yields are an error (HistFitter used 0; set
  `MissingYields: zero`). Signal points without trees are skipped
  (`skipped_points.txt`). Negative MC yields are set to 1e-6.
* **Data-driven stat. uncertainty larger than the yield**: the down variation
  y − s would be ≤ 0 (HistFactory clamps the total at 0); fit2l3l uses y²/(y+s) when
  y − s < 0.1·y. Bins with zero yield get no variation (as in HistFitter).
* **Upper limits** default to a HistFitter-like linear scan (range found automatically,
  log-CLs interpolation); `UpperLimit.Scan: toms748` for pyhf's root finding.
* **Toys**: CLs and the expected band follow pyhf's `ToyCalculator` conventions
  (toys thrown from the parameters profiled on data). A toy whose fit fails with both
  optimizers is dropped and counted (warning) instead of aborting the job. With
  `SaveToys`, toys are thrown at mu = 1, at the theory-varied mu and at the
  `UpperLimit.Scan` points, so CLs, theory lines and limits all come from the merged
  toys.

## Development

```bash
pip install -e '.[test]'
pytest -q            # builds toy ntuples and workspaces, runs all fit types (~2 min)
```

Layout:
```
fit2l3l/
  analysis/base.py        Region, Sample, TreeSyst, WeightSyst, TheorySyst, Analysis
  analysis/ss3l/          SS/3L definitions (regions, samples, systematics, signals, plotting)
  workspace/              composer (YAML -> cabinetry config), templates (cabinetry Router),
                          expressions (C++ cuts), postprocess, builder (ws.json/patchset.json)
  fitting/                runner, bkgonly, discovery, exclusion, toys, tables, batch, common
  plotting/               datamc, contours, style
  cli/                    fit2l3l command, toy input generator
configs/workspace|fits/   example YAMLs
share/                    grids, cross-sections, hard-coded yields, theory uncertainties, HEPData
tests/                    pytest suite (toy inputs)
```
