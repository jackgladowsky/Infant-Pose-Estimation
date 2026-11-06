"""Build/check an allowlisted source+evidence ZIP. Never stages, commits or pushes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "FiDIP_Reproduction.zip"
MANIFEST = HERE / "source_manifest.json"
# Original bytes, not retyped metrics. No raw annotations, images or checkpoint tensors.
EVIDENCE = [
    "data/README.md", "data/validate_datasets.py", "data/archive-manifest.json",
    "data/image-manifest.local.json", "data/validation-report.local.json",
    "data/explorer-validation/2026-10-28/job-11449533-summary.json",
    "models/validation/checkpoint-11450222.json",
    "output/baseline-test100-11450265/metrics.json",
    "output/baseline-test100-11450265/predictions.json",
    "log/finetune-evidence-2026-11-01/train-audit-11553238.out",
    "log/finetune-evidence-2026-11-01/train-pilot-11553500.out",
    "log/finetune-evidence-2026-11-01/finetune-accounting-20261101.txt",
    "output/finetune-pilot-11553500/protocol.json",
    "output/finetune-pilot-11553500/pilot-report.json",
    "output/small-screen-11553889/results.json", "output/small-screen-11553889/protocol.json",
    "output/small-screen-11553889/head/protocol.json",
    "output/small-screen-11553889/stage4_head/protocol.json",
    "output/bounded-diagnostics-11556720/results.json",
    "output/bounded-diagnostics-11556720/protocol.json",
    "log/baseline-evidence-11450265/baseline-11450265.out",
    "log/baseline-evidence-11450265/checkpoint-11450222.out",
    "log/small-screen-evidence-2026-11-01/screen-prep-11553804.out",
    "log/small-screen-evidence-2026-11-01/small-screen-11553889.out",
    "log/small-screen-evidence-2026-11-01/small-screen-accounting-20261101.txt",
    "log/small-screen-evidence-2026-11-01/small-screen-profile-11553804.json",
    "log/bounded-diagnostics-evidence-2026-11-02/bounded-analysis-11556695.json",
    "log/bounded-diagnostics-evidence-2026-11-02/diag-analysis-11556695.out",
    "log/bounded-diagnostics-evidence-2026-11-02/two-diagnostics-11556720.out",
    "log/bounded-diagnostics-evidence-2026-11-02/bounded-diagnostics-accounting-20261102.txt",
]
for mode in ("original_flip_on", "original_flip_off", "head", "stage4_head"):
    EVIDENCE.append(f"output/small-screen-11553889/{mode}/results/keypoints_validate_infant_results_0.json")
for mode in ("original_control", "trained_control", "BN_buffers_restored", "three_crop_TTA"):
    EVIDENCE.append(f"output/bounded-diagnostics-11556720/{mode}/results/keypoints_validate_infant_results_0.json")
CODE_SUFFIXES = {".py", ".pyx", ".cu", ".hpp", ".h", ".yaml", ".yml", ".sh", ".sbatch"}
ROOT_FILES = [
    "README.md", "AGENT.md", "AGENTS.md", "SKILL.md", "ENVIRONMENT.md", "MANIFEST.md",
    ".gitignore", ".python-version", ".env.example", "pyproject.toml", "uv.lock", "fidip_env.yml",
    "FiDIP_Report.pdf",
]


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


def collect_files():
    paths = {ROOT / name for name in ROOT_FILES + EVIDENCE}
    for dirname in ("lib", "tools", "tests", "experiments"):
        for path in (ROOT / dirname).rglob("*"):
            if path.is_file() and (path.suffix in CODE_SUFFIXES or path.name == "Makefile"):
                paths.add(path)
    for dirname in ("reproducibility", "docs"):
        for path in (ROOT / dirname).rglob("*"):
            rel = path.relative_to(ROOT / dirname)
            if any(part in ("build", "previews", "__pycache__") for part in rel.parts):
                continue
            if path.name in {"FINAL_REVIEW.md", "unit-test-results.txt", "validation-summary.txt"}:
                continue
            if path.is_file() and path.suffix in CODE_SUFFIXES | {".md", ".txt", ".tex", ".bib", ".pdf", ".png", ".json"}:
                if path != MANIFEST:
                    paths.add(path)
    result = {}
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symlinked package input: {path}")
        name = path.relative_to(ROOT).as_posix()
        if path.suffix in {".so", ".pth", ".pt", ".ckpt", ".jpg", ".jpeg"}:
            raise ValueError(f"Forbidden binary/private payload: {name}")
        contents = path.read_bytes()
        # Fail closed on common concrete credentials; print only the path, never a value.
        if re.search(rb"(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)", contents):
            raise ValueError(f"Possible secret in package input; inspect privately: {name}")
        result[name] = contents
    return result


def zip_write(bundle, name, contents):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    bundle.writestr(info, contents)


def check():
    registry = json.loads(MANIFEST.read_text())
    with zipfile.ZipFile(PACKAGE) as bundle:
        assert bundle.testzip() is None
        expected = set(registry["files"]) | {"reproducibility/source_manifest.json"}
        assert set(bundle.namelist()) == expected
        assert bundle.read("reproducibility/source_manifest.json") == MANIFEST.read_bytes()
        for name, entry in registry["files"].items():
            assert digest(bundle.read(name)) == entry["sha256"], name
        assert {"AGENT.md", "SKILL.md", "MANIFEST.md", "FiDIP_Report.pdf"} <= expected
    actual = collect_files()
    assert set(actual) == set(registry["files"]), "Package input inventory changed; rebuild"
    for name, contents in actual.items():
        assert digest(contents) == registry["files"][name]["sha256"], f"Stale package: {name}"
    print(f"PASS: {len(registry['files'])} allowlisted files; archive integrity and current-source hashes match")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    files = collect_files()
    registry = {
        "schema_version": 1,
        "base_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
                         if (ROOT / ".git").exists() else "See original manifest; no Git metadata in archive",
        "scope": "Local review/submission bundle, not a Git commit or push. Data archives, raw labels, photographs, weights, caches and compiled binaries excluded.",
        "runtime_portability_changes_since_scored_runs": [
            "analyze_bounded_diagnostics.py and its sbatch accept a new screen directory/checkpoint digest/control AP",
            "run_bounded_diagnostics.py reads those paths/control AP from the newly pinned analysis",
            "These path/control parameters retain historical defaults; no inference or metric mathematics changed. No new cluster rerun claimed.",
        ],
        "files": {name: {"sha256": digest(contents), "bytes": len(contents)} for name, contents in files.items()},
    }
    MANIFEST.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(PACKAGE, "w") as bundle:
        for name, contents in files.items():
            zip_write(bundle, name, contents)
        zip_write(bundle, "reproducibility/source_manifest.json", MANIFEST.read_bytes())
    check()
    print(f"Created {PACKAGE.relative_to(ROOT)} ({PACKAGE.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
