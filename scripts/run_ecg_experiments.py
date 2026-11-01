"""Windows-safe terminal entry point. Only run-suite/preflight execute target models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description="PTB-XL benchmark-aligned school suite for Line")
    parser.add_argument("command", choices=("plan", "preflight", "run-suite", "collect", "summarize"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ptbxl_experiments.json")
    parser.add_argument("--release-dir", type=Path, default=ROOT / "data/processed/ptbxl_v1.0.3_500hz_minimal")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/ptbxl/1.0.3")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "artifacts/school_run")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--allow-download", action="store_true", help="Allow verified official100 downloads on target")
    args = parser.parse_args(argv)
    from ptbxl_experiments import suite
    try:
        if args.command == "plan":
            print(json.dumps(suite.plan(args.config), ensure_ascii=False, indent=2))
        elif args.command == "run-suite":
            return suite.run_suite(args.config, args.release_dir, args.raw_dir, args.work_dir,
                                   device=args.device, allow_download=args.allow_download)
        else:
            args.work_dir.mkdir(parents=True, exist_ok=True)
            with suite.SuiteLock(args.work_dir / "suite.lock"):
                if args.command == "preflight":
                    from ptbxl_experiments.protocol import load_protocol
                    report = suite.preflight(load_protocol(args.config), args.release_dir, args.raw_dir,
                                             args.work_dir, device=args.device)
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                elif args.command == "collect":
                    print(suite.collect_results(args.work_dir))
                elif args.command == "summarize":
                    print(suite.summarize(args.work_dir))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
