"""Lightweight packaging/asset guards; no model inference or source dataset needed."""
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility"))
from prepare_data import safe_extract, link_profile, LINKS


class AssetPreparationTests(unittest.TestCase):
    def archive(self, folder, name):
        path = folder / "test.zip"
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr(name, "payload")
        return path

    def test_safe_extraction_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            path = self.archive(folder, "syrip/a.txt")
            target = folder / "raw"
            safe_extract(path, target)
            self.assertEqual((target / "syrip/a.txt").read_text(), "payload")
            with self.assertRaises(FileExistsError):
                safe_extract(path, target)

    def test_traversal_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            path = self.archive(folder, "../escape")
            with self.assertRaises(ValueError):
                safe_extract(path, folder / "raw")
            self.assertFalse((folder / "raw").exists())

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); path = folder / "test.zip"
            with zipfile.ZipFile(path, "w") as bundle:
                info = zipfile.ZipInfo("link")
                info.external_attr = 0o120777 << 16
                bundle.writestr(info, "../escape")
            with self.assertRaises(ValueError):
                safe_extract(path, folder / "raw")

    def test_case_collision_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); path = folder / "test.zip"
            with zipfile.ZipFile(path, "w") as bundle:
                bundle.writestr("A.txt", "1"); bundle.writestr("a.txt", "2")
            with self.assertRaises(ValueError):
                safe_extract(path, folder / "raw")

    def test_regular_data_not_replaced_by_link(self):
        with tempfile.TemporaryDirectory() as name:
            data = Path(name)
            path = data / "syrip" / next(iter(LINKS))
            path.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                link_profile(data)


class PortabilityTests(unittest.TestCase):
    def test_new_diagnostic_arguments_parse_before_allocation_guard(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import analyze_bounded_diagnostics as diagnostic
        args = ["analyze", "--screen-dir", "output/small-screen-123", "--trained-sha256",
                "a" * 64, "--expected-trained-ap", "93.2"]
        with patch.object(sys, "argv", args), patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "compute allocation"):
                diagnostic.main()

    def test_screen_outside_checkout_is_rejected_before_data_reads(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import analyze_bounded_diagnostics as diagnostic
        with patch.object(sys, "argv", ["analyze", "--screen-dir", "/outside-checkout"]), \
                patch.dict("os.environ", {"SLURM_JOB_ID": "test"}):
            with self.assertRaises(ValueError):
                diagnostic.main()

    def test_recorded_results_check_is_standalone(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "reproducibility/check_results.py")],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("PASS: all six AP results", result.stdout)


if __name__ == "__main__":
    unittest.main()
