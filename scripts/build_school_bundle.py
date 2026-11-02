"""Create a portable, allowlisted ZIP from current working-tree files (no git dependency)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import re
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TEXT_PATHS = (
    "README.md", ".gitignore", ".gitattributes", "requirements-ptbxl.txt", "requirements-ecg-train.txt",
    "configs/ptbxl_experiments.json", "configs/ptbxl_cohort_contract.json", "docs/RUN_SCHOOL.vi.md",
    "configs/official100_recovery_v1.json", "docs/RECOVER_OFFICIAL100.vi.md",
    "reports/PTBXL_PIPELINE_VERIFICATION.vi.md", "reports/PTBXL_PIPELINE_INDEPENDENT_REVIEW.vi.md",
    ".planning/terminal-benchmark/SPEC.md", ".planning/terminal-benchmark/CONTRACT.md",
    ".planning/terminal-benchmark/RESEARCH.md", ".planning/terminal-benchmark/BOOTSTRAP.md",
    "scripts/run_ecg_experiments.py", "scripts/bootstrap_school.py",
    "scripts/recover_official100.py", "tests/test_official100_recovery.py", ".planning/official100-recovery/SPEC.md",
    "scripts/build_school_bundle.py", "scripts/run_ptbxl.py",
    "scripts/windows/Setup-School.ps1", "scripts/windows/Run-School.ps1", "scripts/windows/Bootstrap-School.ps1",
    "notebooks/ptbxl/README.vi.md", "tests/test_school_bundle.py", "tests/test_bootstrap.py",
)
PATTERNS = ("ptbxl_experiments/*.py", "ptbxl_pipeline/*.py", "tests/test_experiment_*.py", "notebooks/ptbxl/*.ipynb")
DATA_ROOTS = ("data/raw/ptbxl/1.0.3", "data/processed/ptbxl_v1.0.3_500hz_minimal")
RAW_TOP_LEVEL = {"LICENSE.txt", "ptbxl_database.csv", "scp_statements.csv", "SHA256SUMS.txt", "source_verification.json"}
RELEASE_TOP_LEVEL = {"audit.json", "DATASET_CARD.vi.md", "metadata.csv", "normalization.json", "pipeline.json",
                     "records.csv", "train.csv", "validation.csv", "test.csv"}
FORBIDDEN_PARTS = {".git", ".venv", ".venv-train", "__pycache__", "node_modules", "artifacts", "cache", "secrets"}


def _safe(root, path, checked_directories=None):
    relative = path.relative_to(root)
    if any(p.lower() in FORBIDDEN_PARTS or p.lower().startswith(".env") or p.lower().endswith((".pem", ".key"))
           for p in relative.parts):
        raise ValueError(f"Forbidden bundle path: {relative}")
    if ".." in relative.parts:
        raise ValueError(f"Traversal is forbidden: {relative}")
    checked_directories = checked_directories if checked_directories is not None else {root}
    cursor = path
    while cursor != root and cursor not in checked_directories:
        if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
            raise ValueError(f"Symlinks/junctions are forbidden in bundle: {relative}")
        if cursor != path:
            checked_directories.add(cursor)
        cursor = cursor.parent
    # root is resolved once; all lexical ancestors have now been checked for links.
    # Cache directories to avoid millions of repeated Windows reparse-point stats.
    return relative.as_posix()


def bundle_files(root, include_data):
    root = Path(root).resolve()
    files = {root / name for name in TEXT_PATHS if (root / name).is_file()}
    for pattern in PATTERNS:
        files.update(p for p in root.glob(pattern) if p.is_file())
    if include_data:
        for name in DATA_ROOTS:
            base = root / name
            if not base.is_dir():
                raise FileNotFoundError(f"Required data root missing: {base}")
            for path in base.rglob("*"):
                if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                    raise ValueError(f"Reject linked source: {path}")
                if path.is_file():
                    relative = path.relative_to(base).as_posix()
                    if name == DATA_ROOTS[0]:
                        approved = relative in RAW_TOP_LEVEL or re.fullmatch(r"records(100|500)/[0-9]{5}/[0-9]{5}_(lr|hr)\.(hea|dat)", relative)
                    else:
                        approved = relative in RELEASE_TOP_LEVEL or re.fullmatch(r"signals/[0-9]{5}\.npy", relative) or re.fullmatch(r"sidecars/[0-9]{5}\.json", relative)
                    if not approved:
                        raise ValueError(f"Unexpected file in public data root: {path}")
                    files.add(path)
    if not files:
        raise ValueError("No approved source files found")
    approved = []
    checked_directories = {root}
    for index, path in enumerate(sorted(files), 1):
        approved.append((path, _safe(root, path, checked_directories)))
        if index % 10000 == 0:
            print(f"allowlist {index}/{len(files)} paths checked", flush=True)
    return approved


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle(root, output, *, include_data=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    checksum = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or checksum.exists():
        raise FileExistsError(f"Will not overwrite bundle/checksum: {output}")
    files = bundle_files(root, include_data)
    if output in {p for p, _ in files}:
        raise ValueError("Output cannot be an input")
    for named in DATA_ROOTS:
        if output.is_relative_to(root / named):
            raise ValueError("Output cannot be inside source data")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "owner": "Line", "include_data": bool(include_data), "files": {},
                "note": "Working-tree snapshot; model/GPU validation runs on school computer only."}
    total = sum(path.stat().st_size for path, _ in files)
    copied = 0
    try:
        with zipfile.ZipFile(temp, "x", allowZip64=True) as archive:
            for index, (path, relative) in enumerate(files, 1):
                digest, size = hashlib.sha256(), 0
                info = zipfile.ZipInfo(relative)
                info.compress_type = zipfile.ZIP_STORED if relative.startswith("data/") else zipfile.ZIP_DEFLATED
                with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as destination:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                manifest["files"][relative] = {"sha256": digest.hexdigest(), "size": size}
                copied += size
                if index % 500 == 0 or index == len(files):
                    print(f"bundle {index}/{len(files)} files; {copied/1024**3:.2f}/{total/1024**3:.2f} GiB", flush=True)
            archive.writestr("TRANSFER_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        # Rename without replacement, even if another process created output while copying.
        if output.exists():
            raise FileExistsError(output)
        if os.name == "nt":
            os.rename(temp, output)
        else:
            os.link(temp, output)
            temp.unlink()
        print("Hashing final ZIP ...", flush=True)
        with checksum.open("x", encoding="ascii") as handle:
            handle.write(f"{_sha256(output)}  {output.name}\n")
    finally:
        if temp.exists():
            temp.unlink()
    return {"output": str(output), "checksum": str(checksum), "files": len(files), "bytes": output.stat().st_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-data", action="store_true")
    args = parser.parse_args(argv)
    result = build_bundle(args.root, args.output, include_data=args.include_data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
