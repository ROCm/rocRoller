################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

"""Run a benchmark suite."""

import argparse
import datetime
import os
import subprocess
import importlib.util

from dataclasses import fields
from itertools import chain
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

import rrperf


def submit_directory(suite: str, wrkdir: Path, ptsdir: Path) -> None:
    """Consolidate performance data and submit it to SOMEWHERE.

    Performance data is read from .yaml files in the work directory
    for the given suite.  Consolidated data is written to a SOMEWHERE
    directory and submitted.
    """
    results = []
    for jpath in wrkdir.glob(f"{suite}-*.yaml"):
        results.extend(yaml.load(jpath.read_text()))
    df = pd.DataFrame(results)
    df.to_csv(f"{str(ptsdir)}/{suite}-benchmark.csv", index=False)
    # TODO: add call to SOMEWHERE to submit


def from_token(token: str):
    yield rrperf.problems.upcast_to_run(eval(token, rrperf.problems.__dict__))


def run_problems(
    generator, build_dir: Path, work_dir: Path, env: Dict[str, str]
) -> bool:

    SOLUTION_NOT_SUPPORTED_ON_ARCH = 3

    already_run = set()
    result = True
    failed = []

    for i, problem in enumerate(generator):
        if filter is not None:
            pass

        if problem in already_run:
            continue

        yaml = (work_dir / f"{problem.group}-{i:06d}.yaml").resolve()
        problem.set_output(yaml)
        cmd = problem.command()
        scmd = " ".join(cmd)
        log = yaml.with_suffix(".log")
        rr_env = {k: str(v) for k, v in env.items() if k.startswith("ROC")}
        rr_env_str = " ".join([f"{k}={v}" for k, v in rr_env.items()])

        with log.open("w") as f:
            print(f"# env: {rr_env_str}", file=f, flush=True)
            print(f"# command: {scmd}", file=f, flush=True)
            print(f"# token: {repr(problem)}", file=f, flush=True)
            print("running:")
            print(f"  command: {scmd}")
            print(f"  wrkdir:  {work_dir.resolve()}")
            print(f"  log:     {log.resolve()}")
            p = subprocess.run(cmd, stdout=f, cwd=build_dir, env=env, check=False)
            status = None
            if p.returncode == 0:
                status = "ok"
            elif p.returncode == SOLUTION_NOT_SUPPORTED_ON_ARCH:
                status = "skipped (not supported on " + rrperf.utils.rocm_gfx() + ")"
            else:
                status = "error"
                result = False
                failed.append((i, problem))
            print(f"  status:  {status}", flush=True)

        already_run.add(problem)

    if len(failed) > 0:
        print(f"Failed {len(failed)} problems:")
        for i, problem in failed:
            cmd = list(map(str, problem.command()))
            print(f"{i}: {' '.join(cmd)}")

    return result


def backcast(generator, build_dir):
    """Reconstruct run objects from `generator` into run objects from previous rrperf version."""
    pdef = build_dir.parent / "scripts" / "lib" / "rrperf" / "problems.py"
    spec = importlib.util.spec_from_file_location("problems", str(pdef))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for run in generator:
        className = run.__class__.__name__
        backClass = getattr(module, className, None)
        if backClass is not None:
            backObj = backClass(
                **{f.name: getattr(run, f.name) for f in fields(backClass)}
            )
            yield backObj


def get_args(parser: argparse.ArgumentParser):
    common_args = [
        rrperf.args.rundir,
        rrperf.args.suite,
    ]
    for arg in common_args:
        arg(parser)

    parser.add_argument(
        "--submit",
        help="Submit results to SOMEWHERE.",
        action="store_true",
        default=False,
    )
    parser.add_argument("--token", help="Benchmark token to run.")
    parser.add_argument("--filter", help="Filter benchmarks...")
    parser.add_argument(
        "--rocm_smi",
        default="rocm-smi",
        help="Location of rocm-smi.",
    )
    parser.add_argument(
        "--pin_clocks",
        action="store_true",
        help="Pin clocks before launching benchmark clients.",
    )


def run(args):
    """Run benchmarks!"""
    run_cli(**args.__dict__)


def run_cli(
    token: str = None,
    suite: str = None,
    submit: bool = False,
    filter: str = None,
    rundir: str = None,
    build_dir: str = None,
    rocm_smi: str = "rocm-smi",
    pin_clocks: bool = False,
    recast: bool = False,
    **kwargs,
) -> Tuple[bool, Path]:
    """Run benchmarks!

    Implements the CLI 'run' subcommand.
    """

    if pin_clocks:
        rrperf.rocm_control.pin_clocks(rocm_smi)

    if suite is None and token is None:
        suite = "all_gfx120X" if rrperf.utils.rocm_gfx().startswith("gfx120") else "all"

    generator = rrperf.utils.empty()
    if suite is not None:
        generator = chain(generator, rrperf.utils.load_suite(suite))
    if token is not None:
        generator = chain(generator, from_token(token))
    if recast:
        generator = backcast(generator, build_dir)

    if build_dir is None:
        build_dir = rrperf.utils.get_build_dir()
    else:
        build_dir = Path(build_dir)

    env = dict(os.environ)
    env["ROCROLLER_ENFORCE_GRAPH_CONSTRAINTS"] = "1"
    env["ROCROLLER_AUDIT_CONTROL_TRACERS"] = "1"

    rundir = rrperf.utils.get_work_dir(rundir, build_dir)
    rundir.mkdir(parents=True, exist_ok=True)

    # pts.create_git_info(str(wrkdir / "git-commit.txt"))
    git_commit = rundir / "git-commit.txt"
    try:
        hash = rrperf.git.full_hash(build_dir)
        git_commit.write_text(f"{hash}\n")
    except Exception:
        git_commit.write_text("NO_COMMIT\n")
    # pts.create_specs_info(str(wrkdir / "machine-specs.txt"))
    machine_specs = rundir / "machine-specs.txt"
    machine_specs.write_text(str(rrperf.specs.get_machine_specs(0, rocm_smi)) + "\n")

    timestamp = rundir / "timestamp.txt"
    timestamp.write_text(str(datetime.datetime.now().timestamp()) + "\n")

    result = run_problems(generator, build_dir, rundir, env)

    if submit:
        ptsdir = rundir / "rocRoller"
        ptsdir.mkdir(parents=True)
        # XXX if running single token, suite might be None
        submit_directory(suite, rundir, ptsdir)

    return result, rundir
