"""Target-only acquisition/preparation/run entry point; plan remains offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ptbxl_pipeline import bootstrap


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare an equivalent PTB-XL release on the school machine")
    parser.add_argument("command", choices=("plan", "prepare", "run"))
    for flag, default in (("config", "configs/ptbxl_experiments.json"),
                          ("contract", "configs/ptbxl_cohort_contract.json"),
                          ("raw-dir", "data/raw/ptbxl/1.0.3"),
                          ("release-dir", "data/processed/ptbxl_v1.0.3_500hz_minimal"),
                          ("cache-dir", "data/cache"), ("work-dir", "artifacts/school_run")):
        parser.add_argument(f"--{flag}", type=Path, default=ROOT / default)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    paths = (args.config, args.contract, args.raw_dir, args.release_dir, args.cache_dir, args.work_dir)
    try:
        if args.command == "plan":
            print(json.dumps(bootstrap.plan(*paths), ensure_ascii=False, indent=2))
            return 0
        target = bootstrap.prepare_school(*paths, allow_download=args.allow_download)
        if args.command == "prepare":
            print(f"Data ready for the frozen target protocol: {target}")
            return 0
        from ptbxl_experiments.suite import run_suite
        return run_suite(target, args.release_dir, args.raw_dir, args.work_dir,
                         device="cuda", allow_download=args.allow_download)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Bootstrap {args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
