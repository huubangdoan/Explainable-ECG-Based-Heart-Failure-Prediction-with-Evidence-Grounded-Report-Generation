"""Transfer ZIP fixtures only; no reads from real source dataset."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class BundleTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("bundle", ROOT / "scripts/build_school_bundle.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def fixture(self, root):
        files = {"ptbxl_experiments/__init__.py": "# untracked fixture", "README.md": "demo",
                 ".env": "do-not-copy", ".git/config": "do-not-copy",
                 "ptbxl_experiments/.env": "do-not-copy", "artifacts/x.py": "do-not-copy",
                 "data/raw/ptbxl/1.0.3/SHA256SUMS.txt": "fixture",
                 "data/processed/ptbxl_v1.0.3_500hz_minimal/audit.json": "{}"}
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def test_source_only_excludes_data_and_secrets_and_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.fixture(root)
            output = Path(tmp) / "transfer.zip"
            self.module.build_bundle(root, output, include_data=False)
            with zipfile.ZipFile(output) as z:
                names = z.namelist()
                self.assertIn("ptbxl_experiments/__init__.py", names)
                self.assertFalse(any(n.startswith("data/") or ".env" in n or ".git" in n for n in names))
                manifest = json.loads(z.read("TRANSFER_MANIFEST.json"))
                self.assertEqual(len(manifest["files"]["README.md"]["sha256"]), 64)
            self.assertTrue(output.with_suffix(".zip.sha256").exists())
            with self.assertRaises(FileExistsError):
                self.module.build_bundle(root, output, include_data=False)

    def test_optional_data_included_only_from_named_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.fixture(root)
            (root / "data" / "private.csv").write_text("do-not-copy")
            output = Path(tmp) / "transfer.zip"
            self.module.build_bundle(root, output, include_data=True)
            with zipfile.ZipFile(output) as z:
                self.assertIn("data/processed/ptbxl_v1.0.3_500hz_minimal/audit.json", z.namelist())
                self.assertNotIn("data/private.csv", z.namelist())

    def test_source_only_bootstrap_contains_contract_entrypoints_and_lf_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.fixture(root)
            bootstrap_files = {
                ".gitattributes": b"*.py text eol=lf\n*.json text eol=lf\n",
                "configs/ptbxl_cohort_contract.json": b'{"fixture":true}\n',
                "ptbxl_pipeline/bootstrap.py": b"# synthetic bootstrap\n",
                "ptbxl_pipeline/core.py": b"# exact LF source bytes\nvalue = 1\n",
                "scripts/bootstrap_school.py": b"# synthetic CLI\n",
                "scripts/windows/Bootstrap-School.ps1": b"# synthetic wrapper\n",
                "tests/test_bootstrap.py": b"# synthetic tests\n",
                ".planning/terminal-benchmark/BOOTSTRAP.md": b"synthetic contract\n",
            }
            for name, content in bootstrap_files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            output = Path(tmp) / "source.zip"
            self.module.build_bundle(root, output, include_data=False)
            with zipfile.ZipFile(output) as archive:
                self.assertFalse(any(name.startswith("data/") for name in archive.namelist()))
                for name, content in bootstrap_files.items():
                    self.assertIn(name, archive.namelist())
                    self.assertEqual(archive.read(name), content)

    def test_unexpected_data_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.fixture(root)
            (root / "data/raw/ptbxl/1.0.3/credentials.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Unexpected file"):
                self.module.build_bundle(root, Path(tmp) / "transfer.zip", include_data=True)

    def test_source_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.fixture(root)
            outside = Path(tmp) / "outside.py"
            outside.write_text("do-not-copy")
            link = root / "ptbxl_experiments/linked.py"
            try:
                link.symlink_to(outside)
            except OSError:
                self.skipTest("OS does not allow fixture symlink creation")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                self.module.build_bundle(root, Path(tmp) / "transfer.zip", include_data=False)


if __name__ == "__main__":
    unittest.main()
