"""Download and verify the immutable, public PTB-XL v1.0.3 source.

Intent: accept source bytes only after matching the official checksum manifest.
Status: checksum validation covers metadata and every 500 Hz header/waveform.
Next: rerun ensure_source after an interrupted acquisition; valid raw files stay.
Blockers: network unavailable or an existing source file has a wrong checksum.
Confidence: exercised by offline fixture and interrupted-download regressions.
"""
from __future__ import annotations

import csv
import concurrent.futures
import hashlib
import http.client
import json
import os
import re
import stat
import time
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SOURCE_VERSION = "1.0.3"
SOURCE_URL = "https://physionet.org/content/ptb-xl/1.0.3/"
FILES_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
S3_FILES_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
ARCHIVE_URL = "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
# Hash of the official SHA256SUMS.txt fetched over HTTPS on 2026-10-24.
OFFICIAL_CHECKSUMS_SHA256 = "b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d"
_METADATA = ("ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt")
_CHUNK_SIZE = 1024 * 1024


class SourceIntegrityError(ValueError):
    """Source contents or paths are unsafe, incomplete, or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(name: str) -> PurePosixPath:
    # Reject Windows separators/drives as well as POSIX traversal, on every OS.
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/"))):
        raise SourceIntegrityError(f"Unsafe relative source path: {name!r}")
    return path


def _comparison_path(resolved: Path) -> Path:
    """Treat Win32 extended-length spellings as the same resolved location."""
    if os.name != "nt":
        return resolved
    value = str(resolved)
    if value.upper().startswith("\\\\?\\UNC\\"):
        return Path("\\\\" + value[8:])
    if re.match(r"^\\\\\?\\[A-Za-z]:\\", value):
        return Path(value[4:])
    return resolved


def _destination(root: Path, name: str) -> Path:
    target = root.joinpath(*_relative_path(name).parts)
    resolved_target, resolved_root = target.resolve(), root.resolve()
    # Windows realpath can retain \\?\ when a concurrent mkdir changes the
    # observed error from PATH_NOT_FOUND to FILE_NOT_FOUND between lookups.
    # Normalize only the equivalent drive/UNC spelling after resolving links;
    # actual containment and the direct symbolic-link guard remain enforced.
    if target.is_symlink() or not _comparison_path(resolved_target).is_relative_to(_comparison_path(resolved_root)):
        raise SourceIntegrityError(
            f"Unsafe source destination: {name!r}; resolved target={str(resolved_target)!r}, "
            f"resolved root={str(resolved_root)!r}"
        )
    return target


def _download_with_resume(url: str, destination: Path, *, attempts: int = 3,
                          expected_sha256: str | None = None, opener=None) -> None:
    """Resume .part only when the server confirms its exact byte offset."""
    destination = Path(destination)
    if destination.exists():
        if expected_sha256 and _sha256(destination) != expected_sha256:
            raise SourceIntegrityError(f"Checksum mismatch for existing {destination}; it was not overwritten.")
        return
    opener = opener or urlopen
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.is_symlink():
        raise SourceIntegrityError(f"Unsafe partial download: {partial}")
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Accept-Encoding": "identity", "User-Agent": "Line-PTBXL-pipeline/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            try:
                response = opener(Request(url, headers=headers), timeout=60)
            except HTTPError as exc:
                complete = re.fullmatch(r"bytes \*/(\d+)", exc.headers.get("Content-Range", ""))
                if exc.code == 416 and complete and offset == int(complete.group(1)) and offset > 0:
                    if expected_sha256 and _sha256(partial) != expected_sha256:
                        raise SourceIntegrityError(f"Checksum mismatch for completed partial {destination.name}.")
                    os.replace(partial, destination)
                    return
                raise
            with response:
                status = response.status
                if status == 206:
                    content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not content_range or int(content_range.group(1)) != offset:
                        raise SourceIntegrityError("HTTP Content-Range does not match the partial download; .part preserved.")
                    start, end, total = map(int, content_range.groups())
                    if end < start or end >= total:
                        raise SourceIntegrityError("Invalid HTTP Content-Range boundaries.")
                    expected_response = end - start + 1
                    content_length = response.headers.get("Content-Length")
                    if content_length and int(content_length) != expected_response:
                        raise SourceIntegrityError("Content-Length disagrees with Content-Range.")
                    mode = "ab" if offset else "wb"
                elif status == 200:
                    # A server may ignore Range. Restart instead of appending duplicates.
                    offset = 0
                    mode = "wb"
                    content_length = response.headers.get("Content-Length")
                    expected_response = int(content_length) if content_length else None
                    total = expected_response
                else:
                    raise SourceIntegrityError(f"Unexpected HTTP status {status} for {url}")
                received = 0
                last_progress = time.monotonic()
                with partial.open(mode) as output:
                    while chunk := response.read(_CHUNK_SIZE):
                        output.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_progress >= 15:
                            print(f"Downloading {destination.name}: {(offset + received) / 1e6:.1f} MB", flush=True)
                            last_progress = time.monotonic()
                    output.flush()
                    os.fsync(output.fileno())
                if expected_response is not None and received != expected_response:
                    raise OSError(f"Incomplete download: received {received}, expected {expected_response} bytes; .part retained.")
                if total is not None and partial.stat().st_size != total:
                    # Some servers deliver a bounded subrange. Continue with the next byte.
                    raise OSError(f"Incomplete download: {partial.stat().st_size} of {total} bytes; .part retained.")
            if expected_sha256 and _sha256(partial) != expected_sha256:
                raise SourceIntegrityError(f"Checksum mismatch while downloading {destination.name}; .part retained.")
            os.replace(partial, destination)
            return
        except (OSError, URLError, http.client.IncompleteRead):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def _download_files(raw_dir: Path, names, checksums: dict[str, str], *, workers: int = 16) -> None:
    """Download missing files with bounded connections and per-thread HTTPS reuse."""
    import requests

    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    names = list(names)
    local = threading.local()
    sessions = []
    sessions_lock = threading.Lock()

    class PooledResponse:
        def __init__(self, response):
            self.response = response
            self.status = response.status_code
            self.headers = response.headers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.response.close()

        def read(self, count):
            try:
                return self.response.raw.read(count, decode_content=False)
            except Exception as exc:
                # urllib3 may surface transport errors outside requests' hierarchy.
                raise OSError(f"Download stream interrupted: {exc}") from exc

    def pooled_open(request, timeout):
        if not hasattr(local, "session"):
            local.session = requests.Session()
            with sessions_lock:
                sessions.append(local.session)
        try:
            response = local.session.get(request.full_url, headers=dict(request.header_items()),
                                         stream=True, timeout=(30, timeout))
        except requests.RequestException as exc:
            raise OSError(f"Download connection failed: {exc}") from exc
        if response.status_code >= 400:
            error = HTTPError(request.full_url, response.status_code, response.reason, response.headers, None)
            response.close()
            raise error
        return PooledResponse(response)

    def download_one(name):
        target = _destination(raw_dir, name)
        _download_with_resume(S3_FILES_URL + name, target,
                              expected_sha256=checksums[name], opener=pooled_open)
        return name

    remaining = iter(names)
    completed = 0
    last_progress = time.monotonic()
    print(f"Downloading {len(names):,} PTB-XL files from public AWS; at most {workers} connections...", flush=True)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    pending = set()
    try:
        # Keep only a small bounded queue, not one future per dataset file.
        for _ in range(workers * 2):
            name = next(remaining, None)
            if name is None:
                break
            pending.add(executor.submit(download_one, name))
        while pending:
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()
                completed += 1
                name = next(remaining, None)
                if name is not None:
                    pending.add(executor.submit(download_one, name))
            if time.monotonic() - last_progress >= 15 or completed == len(names):
                print(f"Downloaded and verified {completed:,}/{len(names):,} files.", flush=True)
                last_progress = time.monotonic()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        for session in sessions:
            session.close()


def _load_checksums(path: Path) -> dict[str, str]:
    actual = _sha256(path)
    if actual != OFFICIAL_CHECKSUMS_SHA256:
        raise SourceIntegrityError(
            f"SHA256SUMS.txt is not the pinned official {SOURCE_VERSION} manifest: "
            f"expected {OFFICIAL_CHECKSUMS_SHA256}, got {actual}. Existing file was not overwritten."
        )
    checksums = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
        if not match:
            raise SourceIntegrityError(f"Malformed SHA256SUMS.txt line {line_number}.")
        checksum, name = match.groups()
        name = _relative_path(name).as_posix()
        if name in checksums:
            raise SourceIntegrityError(f"Duplicate checksum entry: {name}")
        checksums[name] = checksum.lower()
    return checksums


def _verify_existing(raw_dir: Path, names, checksums: dict[str, str]) -> tuple[dict, list]:
    verified = {}
    missing = []
    for name in names:
        if name not in checksums:
            raise SourceIntegrityError(f"Official checksum missing for {name}")
        path = _destination(raw_dir, name)
        if not path.exists():
            missing.append(name)
        elif not path.is_file():
            raise SourceIntegrityError(f"Source is not a regular file: {name}")
        else:
            actual = _sha256(path)
            if actual != checksums[name]:
                raise SourceIntegrityError(
                    f"Checksum mismatch for {name}: expected {checksums[name]}, got {actual}. "
                    "Existing source was not overwritten; move the damaged file aside explicitly before retrying."
                )
            verified[name] = actual
    return verified, missing


def _archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members = {}
    folded = set()
    for member in archive.infolist():
        # ZipInfo normalizes backslashes on Windows; validate original bytes too.
        _relative_path(member.orig_filename)
        name = _relative_path(member.filename).as_posix()
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise SourceIntegrityError(f"Unsafe ZIP symbolic link: {member.filename}")
        if name.casefold() in folded:
            raise SourceIntegrityError(f"Unsafe duplicate ZIP path: {member.filename}")
        folded.add(name.casefold())
        if not member.is_dir():
            members[name] = member
    markers = [name for name in members if PurePosixPath(name).name == "ptbxl_database.csv"]
    if len(markers) != 1:
        raise SourceIntegrityError("ZIP must contain exactly one ptbxl_database.csv source root.")
    prefix = markers[0][:-len("ptbxl_database.csv")]
    return {name[len(prefix):]: item for name, item in members.items() if name.startswith(prefix)}


def _extract_required(archive_path: Path, raw_dir: Path, names, checksums: dict[str, str]) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = _archive_members(archive)
        for name in names:
            if name not in members:
                raise SourceIntegrityError(f"Required source file missing in ZIP: {name}")
            target = _destination(raw_dir, name)
            if target.exists():
                # A caller race must never replace already-existing raw bytes.
                _verify_existing(raw_dir, [name], checksums)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            if partial.is_symlink():
                raise SourceIntegrityError(f"Unsafe extraction temporary file: {partial}")
            digest = hashlib.sha256()
            with archive.open(members[name]) as source, partial.open("wb") as output:
                for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
            if digest.hexdigest() != checksums[name]:
                raise SourceIntegrityError(f"Checksum mismatch while extracting {name}; source file was not published.")
            os.replace(partial, target)


def _record_files(metadata: Path) -> tuple[list[str], int]:
    names = []
    record_ids = set()
    bases = set()
    with metadata.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not {"ecg_id", "filename_hr"}.issubset(rows.fieldnames or []):
            raise SourceIntegrityError("ptbxl_database.csv lacks ecg_id or filename_hr.")
        for row in rows:
            base = _relative_path(row["filename_hr"]).as_posix()
            if not base.startswith("records500/") or not base.endswith("_hr"):
                raise SourceIntegrityError(f"Unexpected 500 Hz source record path: {base}")
            record_id = row["ecg_id"]
            if not record_id or record_id in record_ids or base in bases:
                raise SourceIntegrityError(f"Duplicate or empty ECG identity: {record_id!r}, {base}")
            record_ids.add(record_id)
            bases.add(base)
            names.extend((base + ".hea", base + ".dat"))
    if not record_ids:
        raise SourceIntegrityError("ptbxl_database.csv contains no ECG records.")
    return names, len(record_ids)


def ensure_source(raw_dir: Path, cache_dir: Path, *, download: bool = True) -> dict:
    """Acquire then hash every required source file; emit success only at the end.

    Existing damaged source files cause an actionable error, never replacement.
    With download=False, a complete local source or cached ZIP is required.
    """
    raw_dir, cache_dir = Path(raw_dir).resolve(), Path(cache_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_path = raw_dir / "source_verification.json"
    # A previous report cannot certify files that have changed since that run.
    if report_path.exists():
        report_path.unlink()
    checksum_path = raw_dir / "SHA256SUMS.txt"
    if not checksum_path.exists():
        if not download:
            raise FileNotFoundError("Missing SHA256SUMS.txt; rerun ensure_source(..., download=True).")
        _download_with_resume(S3_FILES_URL + "SHA256SUMS.txt", checksum_path,
                              expected_sha256=OFFICIAL_CHECKSUMS_SHA256)
    checksums = _load_checksums(checksum_path)
    _, missing = _verify_existing(raw_dir, _METADATA, checksums)
    archive_path = cache_dir / "ptbxl-1.0.3.zip"

    def extract_missing(names):
        if not names:
            return
        if not archive_path.exists():
            if not download:
                raise FileNotFoundError("Missing source files and cached ZIP; rerun ensure_source(..., download=True).")
            _download_files(raw_dir, names, checksums)
        else:
            _extract_required(archive_path, raw_dir, names, checksums)

    extract_missing(missing)
    _, missing_metadata = _verify_existing(raw_dir, _METADATA, checksums)
    if missing_metadata:
        raise SourceIntegrityError(
            f"Source metadata still missing after acquisition: {', '.join(missing_metadata)}. "
            "Rerun ensure_source after restoring access to the source files."
        )
    names, record_count = _record_files(raw_dir / "ptbxl_database.csv")
    _, missing = _verify_existing(raw_dir, names, checksums)
    extract_missing(missing)
    # Recheck the entire required set, including files validated before download.
    # A missing-file list is an acquisition failure, never a partial success.
    verified, remaining = _verify_existing(raw_dir, [*_METADATA, *names], checksums)
    if remaining:
        raise SourceIntegrityError(
            f"Source incomplete after acquisition: {len(remaining)} required file(s) missing: "
            f"{', '.join(remaining[:5])}. Rerun ensure_source to recover the missing files."
        )
    report = {
        "source_version": SOURCE_VERSION,
        "source_url": SOURCE_URL,
        "public_mirror_url": S3_FILES_URL,
        "verified": True,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": record_count,
        "verified_files": len(verified),
        "checksum_manifest_sha256": _sha256(checksum_path),
        "file_hashes": dict(sorted(verified.items())),
    }
    temporary = report_path.with_suffix(".json.tmp")
    if temporary.is_symlink():
        raise SourceIntegrityError(f"Unsafe verification report temporary file: {temporary}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    return report
