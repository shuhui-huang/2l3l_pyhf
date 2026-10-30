"""Batch submission of exclusion fits / toys (HistFitter ``--batch``).

Writes a job script and a submit description for HTCondor (lxplus) or Slurm, one
job per chunk of signal points (``--points-per-job``) and toy seed.  The jobs run
``fit2l3l fit ... --points <list> [--seed N]`` and write their results into the
usual ``points/`` (and ``toys/``) directories, which ``fit2l3l merge`` combines.
Missing/failed points are listed by ``fit2l3l merge`` (``missing_points.txt``) and
can be resubmitted with ``--points-file``.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shlex
import subprocess
import sys
from typing import List, Optional, Sequence

log = logging.getLogger(__name__)

CONDOR_FLAVOURS = ["espresso", "microcentury", "longlunch", "workday", "tomorrow", "testmatch", "nextweek"]


def _chunks(items: Sequence[str], n: int) -> List[List[str]]:
    n = max(1, int(n))
    return [list(items[i : i + n]) for i in range(0, len(items), n)]


def write_jobs(
    fit_config: str,
    workspace: str,
    points: Sequence[str],
    *,
    fit_name: Optional[str] = None,
    points_per_job: int = 1,
    seeds: Sequence[int] = (),
    system: str = "condor",
    queue: Optional[str] = None,
    memory: Optional[str] = None,
    setup: Optional[str] = None,
    directory: str = "batch",
    extra_args: str = "",
    submit: bool = False,
) -> pathlib.Path:
    directory = pathlib.Path(directory).resolve()
    (directory / "logs").mkdir(parents=True, exist_ok=True)
    repo = pathlib.Path(__file__).resolve().parents[2]
    setup_cmd = setup or (f"source {repo / 'setup.sh'}" if (repo / "setup.sh").exists() else "")
    fit_config = str(pathlib.Path(fit_config).resolve())
    workspace = str(pathlib.Path(workspace).resolve())
    python = sys.executable

    script = directory / "job.sh"
    fit_opt = f"--fit {shlex.quote(fit_name)}" if fit_name else ""
    script.write_text(
        f"""#!/bin/bash
# fit2l3l batch job: $1 = comma separated signal points, $2 = toy seed (optional)
set -e
cd {shlex.quote(str(repo))}
{setup_cmd}
SEED_OPT=""
if [ -n "$2" ]; then SEED_OPT="--seed $2"; fi
{python} -m fit2l3l.cli.main fit {shlex.quote(fit_config)} --workspace {shlex.quote(workspace)} {fit_opt} \\
    --points "$1" $SEED_OPT --no-merge {extra_args}
"""
    )
    os.chmod(script, 0o755)

    jobs = []
    for chunk in _chunks(list(points), points_per_job):
        for seed in (seeds or [None]):
            jobs.append((",".join(chunk), "" if seed is None else str(seed)))
    jobs_file = directory / "jobs.txt"
    jobs_file.write_text("".join((f"{p} {s}" if s else p) + "\n" for p, s in jobs))
    with_seeds = bool(seeds)
    # HTCondor splits "queue a, b from file" lines on commas *and* spaces and gives
    # the remainder to the last variable: put the seed first, the points last
    condor_file = directory / "jobs_condor.txt"
    condor_file.write_text("".join((f"{s} {p}" if s else p) + "\n" for p, s in jobs))

    if system == "condor":
        queue = queue or "workday"
        if queue not in CONDOR_FLAVOURS:
            raise ValueError(f"HTCondor JobFlavour must be one of {CONDOR_FLAVOURS}")
        sub = directory / "submit.sub"
        sub.write_text(
            f"""executable = {script}
arguments  = {"$(points) $(seed)" if with_seeds else "$(points)"}
output     = {directory}/logs/job.$(ClusterId).$(ProcId).out
error      = {directory}/logs/job.$(ClusterId).$(ProcId).err
log        = {directory}/logs/job.$(ClusterId).log
getenv     = True
+JobFlavour = "{queue}"
{('request_memory = ' + memory) if memory else ''}
queue {"seed, points" if with_seeds else "points"} from {condor_file}
"""
        )
        cmd = ["condor_submit", str(sub)]
    elif system == "slurm":
        sub = directory / "submit.sh"
        sub.write_text(
            f"""#!/bin/bash
#SBATCH --array=1-{len(jobs)}
#SBATCH --output={directory}/logs/job_%A_%a.out
#SBATCH --error={directory}/logs/job_%A_%a.err
{('#SBATCH --partition=' + queue) if queue else ''}
{('#SBATCH --mem=' + memory) if memory else ''}
LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {jobs_file})
{script} $LINE
"""
        )
        cmd = ["sbatch", str(sub)]
    elif system == "local":
        sub = directory / "run_all.sh"
        sub.write_text("#!/bin/bash\n" + "".join(f"{script} {p} {s}\n" for p, s in jobs))
        os.chmod(sub, 0o755)
        cmd = ["bash", str(sub)]
    else:
        raise ValueError("system must be condor, slurm or local")
    log.info(f"{len(jobs)} jobs written to {directory} ({sub.name}); submit with: {' '.join(cmd)}")
    if submit:
        subprocess.run(cmd, check=True)
    return sub
