"""Check transferred SyRIP bytes against the Mac audit, in a CPU allocation only."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def normalize_error(message):
    marker = message.find("/data/")
    return message[marker + 1:] if marker >= 0 else message


def main():
    require(os.environ.get("SLURM_JOB_ID"), "Run in a SLURM CPU allocation, not on a login node")
    started = time.perf_counter()
    expected_report = json.loads((DATA / "validation-report.local.json").read_text())
    expected_images = json.loads((DATA / "image-manifest.local.json").read_text())
    archives = json.loads((DATA / "archive-manifest.json").read_text())
    for entry in archives.values():
        path = ROOT / entry["archive"]
        phase_started = time.perf_counter()
        require(path.stat().st_size == entry["bytes"], f"Archive size mismatch: {path}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        require(actual == entry["sha256"], f"Archive SHA-256 mismatch: {path}")
        log(f"ARCHIVE_PASS name={path.name} sha256={actual} elapsed_s={time.perf_counter() - phase_started:.3f}")

    report_path = DATA / "validation-report.json"
    old_report = report_path.read_bytes() if report_path.exists() else None
    # The data auditor exits 1 for the already-documented training caveats. It must still
    # write a fresh report, and every payload/annotation must match the source audit.
    result = subprocess.run([sys.executable, "-B", str(DATA / "validate_datasets.py")], cwd=ROOT)
    require(result.returncode in (0, 1), f"Data auditor crashed: exit {result.returncode}")
    require(report_path.exists(), "Data auditor did not create its report")
    new_report = report_path.read_bytes()
    require(new_report != old_report, "Data auditor did not refresh the transferred report")
    actual_report = json.loads(new_report)
    actual_images = json.loads((DATA / "image-manifest.json").read_text())
    require(set(actual_images) == set(expected_images), "Extracted image inventory differs from the Mac audit")
    for name, expected in expected_images.items():
        actual = actual_images[name]
        require("error" not in actual, f"Unreadable transferred image: {name}")
        for key in ("sha256", "width", "height"):
            require(actual[key] == expected[key], f"Image {key} mismatch: {name}")
    for group in ("source_annotations", "working_splits"):
        require(set(actual_report[group]) == set(expected_report[group]), f"Annotation inventory mismatch: {group}")
        for name, expected in expected_report[group].items():
            require(actual_report[group][name]["sha256"] == expected["sha256"], f"Annotation hash mismatch: {name}")
    for key in ("baseline_test100_ready", "training_ready_without_review", "cross_split_byte_identical_payloads"):
        require(actual_report[key] == expected_report[key], f"Readiness/overlap changed after transfer: {key}")
    require({normalize_error(e) for e in actual_report["errors"]} ==
            {normalize_error(e) for e in expected_report["errors"]}, "Unexpected dataset errors after transfer")
    require(actual_report["baseline_test100_ready"], "Original Test100 assets did not pass validation")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "job_id": os.environ["SLURM_JOB_ID"],
        "data_root": str(DATA), "archive_hashes_match": True,
        "image_hashes_and_dimensions_matched": len(actual_images),
        "annotation_hashes_match": True, "baseline_test100_ready": True,
        "training_ready_without_review": actual_report["training_ready_without_review"],
        "known_training_errors": actual_report["errors"],
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    (DATA / "explorer-transfer-validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"PASS: byte-identical dataset transfer and baseline Test100 assets ({len(actual_images)} images checked)")
    if not actual_report["training_ready_without_review"]:
        log("CAUTION: original training dimension/leakage issues remain; this is NOT training approval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
