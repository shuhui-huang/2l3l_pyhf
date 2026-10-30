"""Command line interface: ``fit2l3l <command> ...``.

Commands
--------
build       workspace YAML -> cabinetry -> ws.json (+ patchset.json for exclusion)
fit         fit YAML -> pyhf fits on ws.json (bkg-only / discovery / exclusion)
run         build + fit in one go
merge       merge per-point exclusion results (and toys) -> hypotest_list.json
contours    exclusion contours / 1D limits from hypotest_list.json
batch       write (and submit) HTCondor / Slurm jobs for exclusion fits or toys
inspect     print a summary of a workspace
toy-inputs  create toy ntuples in the HistFitter input format (testing)
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import Any, Dict, List

import yaml

from fit2l3l.utils import as_list, load_yaml, setup_logging

log = logging.getLogger("fit2l3l")


def _parse_set(items: List[str]) -> Dict[str, Any]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        out[key] = yaml.safe_load(value)
    return out


def _points(args) -> Any:
    points = as_list(args.points) if getattr(args, "points", None) else None
    if getattr(args, "points_file", None):
        points = (points or []) + [l.strip() for l in pathlib.Path(args.points_file).read_text().splitlines() if l.strip()]
    return points


def cmd_build(args) -> None:
    from fit2l3l.workspace.builder import build_workspace

    outputs = build_workspace(
        args.config,
        overrides=_parse_set(args.set),
        points=_points(args),
        n_workers=args.jobs,
        keep_histograms=not args.no_keep_histograms,
        write_full_workspaces=args.full_workspaces,
    )
    for key, path in outputs.items():
        print(f"{key:18s} {path}")


def _fit_overrides(args) -> Dict[str, Any]:
    over = _parse_set(getattr(args, "set", None))
    points = _points(args)
    if points:
        over["Points"] = points
    if getattr(args, "seed", None) is not None:
        over["Seed"] = args.seed
    if getattr(args, "toys", None):
        over["Calculator"] = "toybased"
        over["NToys"] = args.toys
    if getattr(args, "jobs", None):
        over["Jobs"] = args.jobs
    return over


def cmd_fit(args) -> None:
    from fit2l3l.fitting.runner import run_fits

    results = run_fits(args.config, workspace=args.workspace, only=as_list(args.fit) or None,
                       overrides=_fit_overrides(args), merge=not args.no_merge)
    print(yaml.safe_dump(_jsonable(results), sort_keys=False))


def cmd_run(args) -> None:
    from fit2l3l.fitting.runner import run_fits
    from fit2l3l.workspace.builder import build_workspace

    outputs = build_workspace(args.workspace_config, overrides=_parse_set(args.set_ws), n_workers=args.jobs or 1)
    results = run_fits(args.fit_config, workspace=str(outputs["workspace"].parent), overrides=_fit_overrides(args))
    print(yaml.safe_dump(_jsonable(results), sort_keys=False))


def cmd_merge(args) -> None:
    from fit2l3l.fitting import common, exclusion

    ctx = common.load_context(args.workspace) if args.workspace else None
    points = exclusion.merge(pathlib.Path(args.directory), ctx)
    print(f"{len(points)} points merged into {pathlib.Path(args.directory) / 'hypotest_list.json'}")


def cmd_contours(args) -> None:
    from fit2l3l.fitting import common
    from fit2l3l.plotting import contours

    ctx = common.load_context(args.workspace)
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_yaml(args.config).get("Contours", {}) or {}
    for key, value in (("Model", args.model), ("Interpolation", args.interpolation), ("XKey", args.x_key), ("YKey", args.y_key)):
        if value:
            cfg[key] = value
    if args.draw_cls:
        cfg["DrawCLs"] = True
    if args.draw_cls_exp:
        cfg["DrawCLsExp"] = True
    if args.draw_xs:
        cfg["DrawXS"] = True
    if args.one_d:
        cfg["OneD"] = True
    if args.best:
        cfg["BestSR"] = dict(item.split("=", 1) for item in args.best)
        if args.label:
            cfg["Label"] = args.label
    hypotest = pathlib.Path(args.hypotest_list)
    outdir = pathlib.Path(args.output) if args.output else hypotest.parent / "plots"
    contours.plot_from_config(cfg, merged_file=hypotest, meta=ctx.meta, outdir=outdir)


def cmd_batch(args) -> None:
    from fit2l3l.fitting import batch, common, exclusion
    from fit2l3l.fitting.runner import load_fit_config

    cfg = load_fit_config(args.config, workspace=args.workspace)
    ctx = common.load_context(cfg["General"]["Workspace"])
    fits = [f for f in cfg["Fits"] if not args.fit or f["Name"] == args.fit]
    if not fits:
        raise SystemExit(f"fit '{args.fit}' not found in {args.config}")
    points = _points(args) or exclusion.select_points(ctx, fits[0].get("Points"))
    seeds: List[int] = []
    if args.seeds:
        lo, _, hi = args.seeds.partition("-")
        seeds = list(range(int(lo), int(hi or lo) + 1))
    batch.write_jobs(
        args.config, cfg["General"]["Workspace"], points, fit_name=fits[0]["Name"], points_per_job=args.points_per_job,
        seeds=seeds, system=args.system, queue=args.queue, memory=args.memory, setup=args.setup,
        directory=args.directory or str(ctx.directory / "batch" / fits[0]["Name"]),
        extra_args=" ".join(
            ([f"--toys {args.toys}"] if args.toys else []) + (["--set SaveToys=true"] if seeds else [])
        ),
        submit=args.submit,
    )


def cmd_inspect(args) -> None:
    import json

    import numpy as np
    import pyhf

    from fit2l3l.fitting import common

    ctx = common.load_context(args.workspace)
    ws = ctx.workspace
    print(f"workspace {ctx.directory / 'ws.json'}  ({ctx.meta.get('fit_type')}, {ctx.meta.get('lumi_fb')} fb-1)")
    model = ws.model(poi_name=None)
    print(f"channels: {len(ws.channels)}, samples: {len(ws.samples)}, parameters: {len(model.config.par_names)}"
          f" ({len(model.config.parameters)} modifiers)")
    obs = {o["name"]: o["data"] for o in ctx.spec["observations"]}
    for channel in ctx.spec["channels"]:
        reg = ctx.region(channel["name"])
        tags = [reg.get("role", "?")] + (["fit"] if reg.get("fit") else ["validation"]) + (["blinded"] if reg.get("blinded") else [])
        signal = set(ctx.meta.get("signal", {}) and [p["name"] for p in ctx.meta["signal"]["points"]])
        total = np.sum([np.sum(s["data"]) for s in channel["samples"]
                        if s["name"] not in signal and not s["name"].startswith("DiscoveryMode_")])
        print(f"\n{channel['name']} [{', '.join(tags)}] variable={reg.get('variable')}  data={np.sum(obs[channel['name']]):.1f}  total bkg={total:.2f}")
        for s in channel["samples"]:
            print(f"   {s['name']:24s} {np.sum(s['data']):10.3f}   modifiers: {len(s['modifiers'])}")
    if ctx.patchset is not None:
        print(f"\npatchset: {len(ctx.patchset.patches)} signal points, labels {ctx.patchset.labels}")


def cmd_toy_inputs(rest: List[str]) -> None:
    from fit2l3l.cli import make_toy_inputs

    make_toy_inputs.main(rest)


def _jsonable(obj):
    import json

    from fit2l3l.utils import to_builtin

    return to_builtin(obj)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fit2l3l", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="count", default=1, help="-v: info (default), -vv: debug")
    parser.add_argument("-q", "--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="build the workspace (cabinetry)")
    p.add_argument("config", help="workspace YAML")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override General.<KEY>, e.g. --set Systematics=NONE")
    p.add_argument("--points", help="comma separated signal points (DSIDs) for exclusion workspaces")
    p.add_argument("--points-file", help="file with one signal point per line")
    p.add_argument("-j", "--jobs", type=int, default=1, help="parallel processes for reading ntuples")
    p.add_argument("--no-keep-histograms", action="store_true", help="delete per-signal-point histograms after use")
    p.add_argument("--full-workspaces", action="store_true", help="also write one full workspace per signal point")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("fit", help="run the fits of a fit YAML (pyhf)")
    p.add_argument("config", help="fit YAML")
    p.add_argument("--workspace", help="workspace directory (overrides General.Workspace)")
    p.add_argument("--fit", action="append", help="run only these fits (by Name)")
    p.add_argument("--points", help="signal points (exclusion)")
    p.add_argument("--points-file", help="file with one signal point per line")
    p.add_argument("--seed", type=int, help="random seed for toys")
    p.add_argument("--toys", type=int, help="use toys (number per job)")
    p.add_argument("-j", "--jobs", type=int, help="parallel processes over signal points")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a fit option, e.g. --set Asimov=true")
    p.add_argument("--no-merge", action="store_true", help="do not merge exclusion results (batch jobs)")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("run", help="build the workspace and run the fits")
    p.add_argument("workspace_config")
    p.add_argument("fit_config")
    p.add_argument("--set-ws", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("-j", "--jobs", type=int)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("merge", help="merge per-point exclusion results and toys")
    p.add_argument("directory", help="fit output directory (containing points/)")
    p.add_argument("--workspace", help="workspace directory (to list missing points)")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("contours", help="exclusion contours / 1D upper limits")
    p.add_argument("hypotest_list", help="hypotest_list.json written by the exclusion fit / merge")
    p.add_argument("--workspace", required=True, help="workspace directory (meta.json)")
    p.add_argument("--config", help="fit YAML with a Contours block")
    p.add_argument("--model", help="plotting model (default: grid name)")
    p.add_argument("--interpolation", choices=["linear", "cubic", "rbf", "nearest"])
    p.add_argument("--x-key")
    p.add_argument("--y-key")
    p.add_argument("--best", action="append", metavar="LABEL=hypotest_list.json", help="combine with other SRs (best expected CLs)")
    p.add_argument("--label", help="label of this result for --best")
    p.add_argument("--draw-cls", action="store_true")
    p.add_argument("--draw-cls-exp", action="store_true")
    p.add_argument("--draw-xs", action="store_true")
    p.add_argument("--one-d", "--1d", dest="one_d", action="store_true", help="upper limit on the cross-section vs mass")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_contours)

    p = sub.add_parser("batch", help="write/submit batch jobs for exclusion fits or toys")
    p.add_argument("config", help="fit YAML")
    p.add_argument("--workspace")
    p.add_argument("--fit", help="fit Name (default: first)")
    p.add_argument("--points")
    p.add_argument("--points-file", help="e.g. missing_points.txt for resubmission")
    p.add_argument("--points-per-job", type=int, default=5)
    p.add_argument("--seeds", help="toy seeds, e.g. 1-20 (one job per seed and chunk)")
    p.add_argument("--toys", type=int, help="toys per job")
    p.add_argument("--system", choices=["condor", "slurm", "local"], default="condor")
    p.add_argument("--queue", default=None, help="HTCondor JobFlavour (default workday) or Slurm partition")
    p.add_argument("--memory")
    p.add_argument("--setup", help="environment setup command run by each job")
    p.add_argument("--directory")
    p.add_argument("--submit", action="store_true")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("inspect", help="summary of a workspace")
    p.add_argument("workspace")
    p.set_defaults(func=cmd_inspect)

    sub.add_parser("toy-inputs", help="create toy ntuples (see fit2l3l toy-inputs --help)", add_help=False)
    return parser


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "toy-inputs":
        setup_logging(1)
        return cmd_toy_inputs(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(0 if args.quiet else args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
