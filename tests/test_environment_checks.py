"""Local harness regressions; these do not compile extensions or require a GPU."""
import contextlib
import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("environment_checks", ROOT / "tools/check_environment.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class CheckHarnessTests(unittest.TestCase):
    def test_timing_reports_success(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertTrue(checks.timed_step("example", lambda: None))
        self.assertIn("STEP_START name=example", output.getvalue())
        self.assertIn("status=PASS elapsed_s=", output.getvalue())

    def test_timing_reports_exception(self):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.assertFalse(checks.timed_step("broken", lambda: 1 / 0))
        self.assertIn("status=FAIL elapsed_s=", output.getvalue())
        self.assertIn("ZeroDivisionError", errors.getvalue())

    def test_cpu_phase_does_not_invoke_cuda(self):
        with mock.patch.object(checks, "cpu_nms_check") as cpu, \
             mock.patch.object(checks, "cuda_check") as cuda, \
             mock.patch.object(checks.faulthandler, "enable"), \
             mock.patch.object(checks.faulthandler, "dump_traceback_later"), \
             mock.patch.object(checks.faulthandler, "cancel_dump_traceback_later"), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checks.main(["--phase", "cpu"]), 0)
        cpu.assert_called_once()
        cuda.assert_not_called()

    def test_failure_does_not_skip_later_checks(self):
        executed = []

        def step(name, action):
            executed.append(name)
            return name != "cpu_nms"

        with mock.patch.object(checks, "timed_step", side_effect=step), \
             mock.patch.object(checks.faulthandler, "enable"), \
             mock.patch.object(checks.faulthandler, "dump_traceback_later"), \
             mock.patch.object(checks.faulthandler, "cancel_dump_traceback_later"), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checks.main(["--phase", "all"]), 1)
        self.assertIn("gpu_nms", executed)
        self.assertIn("models_import", executed)

    def test_nms_cases_include_empty_and_cross_block_inputs(self):
        cases = {name: (boxes, keep) for name, boxes, keep in checks.nms_cases(np)}
        self.assertEqual(cases["empty"][0].shape, (0, 5))
        boxes, expected = cases["cross_64_box_blocks"]
        self.assertEqual(boxes.dtype, np.float32)
        self.assertEqual(len(boxes), 129)
        self.assertNotIn(64, expected)
        self.assertNotIn(128, expected)

    def test_nms_harness_rejects_incorrect_result(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(AssertionError, "NMS case single"):
                checks.check_nms(lambda boxes, threshold: [], np)

    def test_h200_is_rejected_before_any_tensor_allocation(self):
        torch = mock.MagicMock()
        torch.__version__ = "test"
        torch.version.cuda = "12.1"
        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = "NVIDIA H200"
        torch.cuda.get_device_capability.return_value = (9, 0)
        torch.cuda.get_device_properties.return_value.total_memory = 140 * 1024 ** 3
        with mock.patch.object(checks.importlib, "import_module", return_value=torch), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "not approved"):
                checks.cuda_check()
        torch.ones.assert_not_called()

    def test_gpu_script_requires_v100_and_does_not_build(self):
        script = (ROOT / "tools/check_environment.sbatch").read_text()
        self.assertIn("#SBATCH --gres=gpu:v100-sxm2:1", script)
        self.assertNotIn("#SBATCH --gres=gpu:1\n", script)
        self.assertNotIn("build_ext", script)
        self.assertIn("--time=00:05:00", script)

    def test_cpu_build_has_no_gpu_request(self):
        script = (ROOT / "tools/build_environment.sbatch").read_text()
        self.assertNotIn("#SBATCH --gres", script)
        self.assertIn("--phase cpu", script)
        self.assertIn("--cython-c-in-temp", script)

    @unittest.skipUnless(shutil.which("timeout"), "GNU timeout is not installed locally")
    def test_shell_stage_preserves_failure_and_timeout_status(self):
        for command, expected in [("exit 7", 7), ("sleep 5", 124)]:
            with self.subTest(command=command), tempfile.TemporaryDirectory() as directory:
                activate = Path(directory) / ".venv/bin/activate"
                activate.parent.mkdir(parents=True)
                activate.write_text("")
                result = subprocess.run(
                    ["bash", "-c", 'module() { :; }; source "$1"; stage probe 0.1s bash -c "$2"',
                     "test", str(ROOT / "tools/environment_check_common.sh"), command],
                    cwd=directory, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertIn(f"STAGE_END name=probe rc={expected}", result.stdout)
                self.assertIn("JOB_END", result.stdout)


if __name__ == "__main__":
    unittest.main()
