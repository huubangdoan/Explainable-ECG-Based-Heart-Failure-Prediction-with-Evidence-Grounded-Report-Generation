"""Plan, migrate, then resume the certified official100 incident on school only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ptbxl_pipeline import recovery


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recover Line's 18 completed jobs after the certified official100 lead-case repair")
    parser.add_argument("command", choices=("plan", "migrate", "run"))
    for flag, default in (("old-work-dir", "artifacts/school_run"),
                          ("work-dir", "artifacts/school_run_recovered"),
                          ("release-dir", "data/processed/ptbxl_v1.0.3_500hz_minimal"),
                          ("raw-dir", "data/raw/ptbxl/1.0.3"),
                          ("certificate", "configs/official100_recovery_v1.json"),
                          ("contract", "configs/ptbxl_cohort_contract.json")):
        parser.add_argument(f"--{flag}", type=Path, default=ROOT / default)
    parser.add_argument("--allow-download", action="store_true",
                        help="Only run may download genuinely missing official100 source files on school")
    args = parser.parse_args(argv)
    kwargs = dict(old_work_dir=args.old_work_dir, work_dir=args.work_dir, release_dir=args.release_dir,
                  raw_dir=args.raw_dir, certificate_path=args.certificate, contract_path=args.contract)
    try:
        if args.command == "plan":
            print(json.dumps(recovery.plan(**kwargs), indent=2, ensure_ascii=False))
            return 0
        if args.command == "migrate":
            print(f"Recovery verified; continue with this recovery entrypoint's run command: {recovery.migrate(**kwargs)}")
            return 0
        return recovery.run(**kwargs, allow_download=args.allow_download)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Recovery {args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
