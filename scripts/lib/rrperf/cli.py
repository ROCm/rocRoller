"""rocRoller performance tracking suite command line interface."""

import argparse
import rrperf


def main():
    parser = argparse.ArgumentParser(
        description="rocRoller performance tracking suite."
    )
    subparsers = parser.add_subparsers(dest="command")

    run_cmd = subparsers.add_parser("run")
    run_cmd.add_argument("--suite", help="Benchmark suite to run.")
    run_cmd.add_argument(
        "--submit", help="Submit results to SOMEWHERE.", action="store_true", default=False
    )
    run_cmd.add_argument(
        "--working_dir",
        help="Location to run tests and store performance results.",
        default=None,
    )
    run_cmd.add_argument("--token", help="Benchmark token to run.")
    run_cmd.add_argument("--filter", help="Filter benchmarks...")
    run_cmd.add_argument("--rocm_smi", default="rocm-smi", help="Location of rocm-smi.")
    run_cmd.add_argument("--pin_clocks", action="store_true", help="Pin clocks before launching benchmark clients.")

    compare_cmd = subparsers.add_parser("compare")
    compare_cmd.add_argument(
        "directories", nargs="*", help="Output directories to compare."
    )
    compare_cmd.add_argument(
        "--format",
        choices=["md", "html", "email_html"],
        default="md",
        help="Output format.",
    )

    args = parser.parse_args()
    command = {"run": rrperf.run.run, "compare": rrperf.compare.compare}[args.command]
    command(**args.__dict__)
