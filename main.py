"""
ML Statistical Arbitrage Pipeline — Entry Point
Krauss et al. (2017) Replication + Extension

Usage
-----
  python main.py                          # run all steps (skip if outputs exist)
  python main.py --steps gbt ensemble     # run specific steps only
  python main.py --force                  # re-run all steps regardless
  python main.py --force --steps ensemble # force re-run specific steps
  python main.py --dashboard              # launch Streamlit dashboard
  python main.py --list                   # list all steps and their status
"""

import argparse
import subprocess
import sys
import os
import yaml
from pathlib import Path

# ── Resolve project root (this file lives at Projects/) ──────
ROOT = Path(__file__).parent.resolve()


def load_config(path: str = "config.yaml") -> dict:
    cfg_path = ROOT / path
    if not cfg_path.exists():
        sys.exit(f"[ERROR] config file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def outputs_exist(output_list: list) -> bool:
    """Return True only if every output file already exists."""
    return all((ROOT / p).exists() for p in output_list)


def run_step(step: dict, force: bool) -> bool:
    """
    Run a single pipeline step.
    Returns True on success, False on skip or failure.
    """
    name    = step["name"]
    script  = step.get("script")
    outputs = step.get("outputs", [])

    # RF notebook — always skip with a message
    if script is None:
        if outputs_exist(outputs):
            print(f"  [SKIP]  {name:<12} outputs found, skipping (notebook step)")
        else:
            print(f"  [WARN]  {name:<12} script=null — run Code/random_forest.ipynb manually in Jupyter")
        return True

    script_path = ROOT / script
    if not script_path.exists():
        print(f"  [ERROR] {name:<12} script not found: {script_path}")
        return False

    if not force and outputs_exist(outputs):
        print(f"  [SKIP]  {name:<12} all outputs exist")
        return True

    print(f"  [RUN]   {name:<12} -> {script}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"  [FAIL]  {name} exited with code {result.returncode}")
        return False

    print(f"  [DONE]  {name}")
    return True


def list_steps(cfg: dict) -> None:
    """Print status of all pipeline steps."""
    print(f"\n{'Step':<14} {'Status':<12} {'Script'}")
    print("-" * 60)
    for step in cfg["pipeline"]["steps"]:
        name    = step["name"]
        script  = step.get("script", "—")
        outputs = step.get("outputs", [])

        if script is None:
            status = "NOTEBOOK"
        elif outputs_exist(outputs):
            status = "DONE"
        else:
            status = "PENDING"

        label = script if script else "run manually"
        print(f"  {name:<12} {status:<12} {label}")
    print()


def launch_dashboard(cfg: dict) -> None:
    port = cfg.get("dashboard", {}).get("port", 8501)
    dash = ROOT / "dashboard.py"
    if not dash.exists():
        sys.exit(f"[ERROR] dashboard.py not found at {dash}")
    print(f"\nLaunching dashboard on http://localhost:{port} ...")
    subprocess.run(
        ["streamlit", "run", str(dash), f"--server.port={port}"],
        cwd=str(ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML Statistical Arbitrage Pipeline"
    )
    parser.add_argument(
        "--steps", nargs="+", metavar="STEP",
        help="run only these steps (e.g. --steps gbt ensemble)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-run steps even if outputs already exist",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="launch the Streamlit performance dashboard",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="list all steps and their current status, then exit",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="path to config file (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.list:
        list_steps(cfg)
        return

    if args.dashboard:
        launch_dashboard(cfg)
        return

    # ── Select which steps to run ─────────────────────────────
    all_steps = cfg["pipeline"]["steps"]
    if args.steps:
        step_names = {s["name"] for s in all_steps}
        unknown = [s for s in args.steps if s not in step_names]
        if unknown:
            sys.exit(f"[ERROR] Unknown step(s): {unknown}. "
                     f"Valid steps: {sorted(step_names)}")
        steps_to_run = [s for s in all_steps if s["name"] in args.steps]
    else:
        steps_to_run = all_steps

    force = args.force or not cfg["pipeline"].get("skip_if_exists", True)

    # ── Run pipeline ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ML Statistical Arbitrage Pipeline")
    print("=" * 60)
    print(f"  Config : {ROOT / args.config}")
    print(f"  Steps  : {[s['name'] for s in steps_to_run]}")
    print(f"  Force  : {force}")
    print("=" * 60 + "\n")

    failed = []
    for step in steps_to_run:
        ok = run_step(step, force=force)
        if not ok:
            failed.append(step["name"])

    print("\n" + "=" * 60)
    if failed:
        print(f"  FAILED steps: {failed}")
    else:
        print("  All steps completed successfully.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
