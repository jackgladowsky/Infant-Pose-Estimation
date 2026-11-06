"""Safely prepare the two checksum-pinned SyRIP releases in a NEW checkout.

Run only in a CPU compute allocation on Explorer. Does not download assets or alter labels.
The archived auditor and reference metadata are supplied by build_package.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    "SyRIP.zip": ("1d13e816375988be9bbdab8bf5be6562a5b9bfa83684d57ccbfec9dd619e1692", "original"),
    "syrip_for_train.zip": ("1f267878732fb81776c5c7f153bad45d652cadf5bfd83761971390b64d6cee1b", "fidip"),
}
LINKS = {
    "images/train_infant": "../../raw/fidip/syrip/images/train_infant",
    "images/train_pre_infant": "../../raw/fidip/syrip/images/train_pre_infant",
    "images/validate_infant": "../../raw/original/SyRIP/images/validate_infant",
    "annotations/person_keypoints_train_infant.json": "../../raw/fidip/syrip/annotations/person_keypoints_train_infant.json",
    "annotations/person_keypoints_train_pre_infant.json": "../../raw/fidip/syrip/annotations/person_keypoints_train_pre_infant.json",
    "annotations/person_keypoints_validate_infant.json": "../../raw/original/SyRIP/annotations/validate100/person_keypoints_validate_infant.json",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_extract(archive, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing extraction: {destination}")
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        seen = set()
        # Validate EVERY member before creating any output.
        for member in members:
            path = PurePosixPath(member.filename)
            if (not member.filename or path.is_absolute() or ".." in path.parts
                    or "\\" in member.filename or ":" in member.filename
                    or stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1):
                raise ValueError(f"Unsafe or encrypted ZIP member: {member.filename!r}")
            normalized = str(path).casefold()
            if normalized in seen:
                raise ValueError(f"Duplicate/case-colliding ZIP member: {member.filename!r}")
            seen.add(normalized)
        destination.mkdir(parents=True)
        bundle.extractall(destination)  # zipfile verifies member CRC while extracting.


def link_profile(data):
    for name, target in LINKS.items():
        path = data / "syrip" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            if os.readlink(path) != target:
                raise FileExistsError(f"Conflicting link: {path}")
        elif path.exists():
            raise FileExistsError(f"Refusing to replace existing data: {path}")
        else:
            path.symlink_to(target)
        if not path.exists():
            raise FileNotFoundError(f"Broken working-profile link: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="Extract only into absent raw roots")
    parser.add_argument("--audit", action="store_true", help="Run the supplied full auditor/transfer verifier")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use a CPU SLURM allocation; never run full data preparation on a login node")
    data = ROOT / "data"
    # Check both immutable inputs before any extraction; files must be acquired separately.
    for name, (expected, _) in ARCHIVES.items():
        if sha(data / "archives" / name) != expected:
            raise ValueError(f"Archive SHA-256 mismatch: {name}")
    if args.extract:
        for _, subdir in ARCHIVES.values():
            if (data / "raw" / subdir).exists():
                raise FileExistsError(f"Extraction root already exists: {subdir}; omit --extract")
        for name, (_, subdir) in ARCHIVES.items():
            safe_extract(data / "archives" / name, data / "raw" / subdir)
    link_profile(data)
    if args.audit:
        # The transfer wrapper independently compares fresh image/annotation hashes with the
        # supplied reference manifests. Raw auditor exit 1 for known training caveats is expected.
        subprocess.run([sys.executable, "-B", "tools/validate_transferred_dataset.py"],
                       cwd=ROOT, check=True)
        report = json.loads((data / "explorer-transfer-validation.json").read_text())
        if not report["baseline_test100_ready"]:
            raise RuntimeError("Test100 admission failed")
    print("PASS: immutable archive identities and original Test100 working-profile links")


if __name__ == "__main__":
    main()
