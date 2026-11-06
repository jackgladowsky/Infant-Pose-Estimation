"""Timed smoke checks. CPU phase needs no GPU; all/gpu phases need an allocation."""
import argparse
import faulthandler
import importlib
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def timed_step(name, action):
    started = time.perf_counter()
    log(f"STEP_START name={name}")
    try:
        action()
    except Exception:
        traceback.print_exc()
        log(f"STEP_END name={name} status=FAIL elapsed_s={time.perf_counter() - started:.3f}")
        return False
    log(f"STEP_END name={name} status=PASS elapsed_s={time.perf_counter() - started:.3f}")
    return True


def nms_cases(np):
    """Known answers, including empty input and CUDA bitmask-block boundaries."""
    cases = [
        ("empty", [], []),
        ("single", [[0, 0, 10, 10, 0.9]], [0]),
        ("duplicate", [[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8]], [0]),
        ("disjoint_unsorted", [[0, 0, 10, 10, 0.2], [20, 20, 30, 30, 0.9]], [1, 0]),
        ("mixed", [[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8],
                   [20, 20, 30, 30, 0.7]], [0, 2]),
    ]
    for name, boxes, expected in cases:
        yield name, np.asarray(boxes, dtype=np.float32).reshape(-1, 5), expected
    boxes = np.asarray([[20 * i, 0, 20 * i + 10, 10, 0.99 - i * 0.001]
                        for i in range(129)], dtype=np.float32)
    boxes[64, :4] = boxes[0, :4]
    boxes[128, :4] = boxes[63, :4]
    yield "cross_64_box_blocks", boxes, [i for i in range(129) if i not in (64, 128)]


def check_nms(implementation, np, synchronize=None):
    for name, boxes, expected in nms_cases(np):
        started = time.perf_counter()
        log(f"CASE_START name={name} boxes={len(boxes)}")
        if synchronize:
            synchronize()
        actual = list(implementation(boxes, 0.5))
        if synchronize:
            synchronize()
        if actual != expected:
            raise AssertionError(f"NMS case {name}: expected {expected}, got {actual}")
        log(f"CASE_PASS name={name} boxes={len(boxes)} elapsed_s={time.perf_counter() - started:.3f}")


def cpu_nms_check():
    np = importlib.import_module("numpy")
    cpu_nms = importlib.import_module("nms.cpu_nms").cpu_nms
    check_nms(cpu_nms, np)


def cuda_check():
    torch = importlib.import_module("torch")
    log(f"PyTorch={torch.__version__} CUDA_runtime={torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available; use an allocated GPU node")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    log(f"GPU={name} capability={capability} memory_GiB={memory:.1f}")
    if not any(model in name.upper() for model in ("V100", "A100")):
        raise RuntimeError(f"GPU {name!r} is not approved: request V100 or an explicitly chosen A100")
    x = torch.ones((32, 32), device="cuda")
    result = (x @ x).sum().item()
    torch.cuda.synchronize()
    if result != 32768:
        raise AssertionError(f"CUDA matmul returned {result}, expected 32768")


def gpu_nms_check():
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    # Repeat the model guard so this step never bypasses a failed resource check.
    name = torch.cuda.get_device_name(0).upper()
    if not any(model in name for model in ("V100", "A100")):
        raise RuntimeError("Unapproved GPU model; refusing GPU NMS")
    gpu_nms = importlib.import_module("nms.gpu_nms").gpu_nms
    check_nms(lambda boxes, threshold: gpu_nms(boxes, threshold, 0), np,
              synchronize=torch.cuda.synchronize)


def import_check(module, attribute=None):
    loaded = importlib.import_module(module)
    if attribute:
        getattr(loaded, attribute)
    version = getattr(loaded, "__version__", "not reported")
    log(f"IMPORT module={module} version={version}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cpu", "gpu", "all"), default="all")
    args = parser.parse_args(argv)
    log(f"Python={sys.version} executable={sys.executable} phase={args.phase}")
    faulthandler.enable()
    # If a native import/file read stalls, emit the Python stack before the stage timeout.
    faulthandler.dump_traceback_later(60, repeat=True)
    checks = []
    if args.phase in ("gpu", "all"):
        checks.append(("cuda_runtime_and_matmul", cuda_check))
    if args.phase in ("cpu", "all"):
        checks.append(("cpu_nms", cpu_nms_check))
    if args.phase in ("gpu", "all"):
        checks.extend([
            ("gpu_nms", gpu_nms_check),
            ("torchvision_import", lambda: import_check("torchvision")),
            ("dataset_import", lambda: import_check("dataset")),
            ("models_import", lambda: import_check("models")),
            ("tensorboard_import", lambda: import_check("torch.utils.tensorboard", "SummaryWriter")),
        ])
    try:
        # Do not let a CPU-NMS exception hide the independent GPU/import results.
        failures = [name for name, action in checks if not timed_step(name, action)]
    finally:
        faulthandler.cancel_dump_traceback_later()
    if failures:
        log(f"FAIL phase={args.phase} failed_steps={','.join(failures)}")
        return 1
    log(f"PASS phase={args.phase} steps={len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
