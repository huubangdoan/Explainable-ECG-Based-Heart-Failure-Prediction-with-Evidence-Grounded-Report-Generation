"""Execute the six PTB-XL notebooks with a separate kernel for every task."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys


NOTEBOOK_NAMES = (
    "00_chuan_bi_du_lieu.ipynb",
    "01_tao_nhan_va_chia_tap.ipynb",
    "02_tien_xu_ly_va_qc.ipynb",
    "03_chuan_hoa_tu_train.ipynb",
    "04_kiem_dinh_dong_goi.ipynb",
    "05_nap_batch_train_test.ipynb",
)


@contextmanager
def notebook_environment(project_root: Path, raw_dir: Path | None, dataset_dir: Path | None):
    values = {"PTBXL_PROJECT_ROOT": str(project_root)}
    if raw_dir is not None:
        values["PTBXL_RAW_DIR"] = str(raw_dir.resolve())
    if dataset_dir is not None:
        values["PTBXL_OUTPUT_DIR"] = str(dataset_dir.resolve())
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def execute_notebooks(
    project_root: Path,
    executed_dir: Path,
    *,
    kernel_name: str = "python3",
    timeout: int = 3600,
    raw_dir: Path | None = None,
    dataset_dir: Path | None = None,
) -> None:
    import nbformat
    from nbclient import NotebookClient

    project_root = project_root.resolve()
    source_dir = project_root / "notebooks" / "ptbxl"
    executed_dir = executed_dir.resolve()
    if executed_dir == source_dir.resolve():
        raise ValueError("Executed notebooks must not overwrite the source notebooks.")
    missing = [str(source_dir / name) for name in NOTEBOOK_NAMES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing notebooks: " + ", ".join(missing))
    executed_dir.mkdir(parents=True, exist_ok=True)

    # Intent: prove independence between stages and preserve the failing cell.
    # Status: each NotebookClient starts and closes its own kernel.
    # Next: root runs the complete sequence on the official dataset.
    # Blocker: missing packages/data remain visible errors; confidence: tested.
    with notebook_environment(project_root, raw_dir, dataset_dir):
        for index, name in enumerate(NOTEBOOK_NAMES, start=1):
            print(f"[{index}/6] {name}", flush=True)
            notebook = nbformat.read(source_dir / name, as_version=4)
            client = NotebookClient(
                notebook,
                kernel_name=kernel_name,
                timeout=timeout,
                resources={"metadata": {"path": str(project_root)}},
                allow_errors=False,
            )
            destination = executed_dir / name
            try:
                client.execute()
            finally:
                temporary = destination.with_suffix(".ipynb.tmp")
                nbformat.write(notebook, temporary)
                temporary.replace(destination)
            print(f"Saved: {destination}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PTB-XL notebooks 00-05 in six fresh kernels.")
    parser.add_argument("--execute-notebooks", action="store_true", help="Execute every notebook in order; stop on the first error.")
    parser.add_argument("--kernel-name", default="python3", help="Installed Jupyter kernel name (recommended: ptbxl).")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--raw-dir", type=Path, help="Override PTBXL_RAW_DIR.")
    parser.add_argument("--output-dir", type=Path, help="Override the prepared dataset directory, PTBXL_OUTPUT_DIR.")
    parser.add_argument("--executed-dir", type=Path, help="Destination for executed notebook copies.")
    parser.add_argument("--timeout", type=int, default=3600, help="Maximum seconds per code cell; increase for a slow download.")
    args = parser.parse_args(argv)
    if not args.execute_notebooks:
        parser.print_help()
        return 0
    if args.timeout <= 0:
        parser.error("--timeout must be a positive number of seconds")
    root = args.project_root.resolve()
    if not (root / "ptbxl_pipeline").is_dir() or not (root / "requirements-ptbxl.txt").is_file():
        parser.error("--project-root must contain ptbxl_pipeline and requirements-ptbxl.txt")
    executed_dir = args.executed_dir or root / "artifacts" / "executed_notebooks"
    execute_notebooks(root, executed_dir, kernel_name=args.kernel_name, timeout=args.timeout, raw_dir=args.raw_dir, dataset_dir=args.output_dir)
    print("Completed all six notebooks. Inspect audit.json before using the release.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
