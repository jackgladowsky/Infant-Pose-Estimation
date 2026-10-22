# uv environment on Explorer

This is a Linux x86-64 CUDA environment, not a macOS environment. Original Conda recipe is retained
for reference; `pyproject.toml` defines the smaller uv-managed training/evaluation environment.
Synthetic-generation dependencies are separate and not installed here.

## Setup

uv is installed at `/home/gladowsky.j/.local/bin/uv` (0.12.17 at setup time).
This is a personal-account executable, not a system-wide install. It was installed without modifying
shell startup files. No uv-specific institutional endorsement was found; see `SKILL.md` for official
RC policy links and the distinction between user-local tools and large research environments.

**Further installation/download/move work is paused until storage paths are agreed with Jack.**
Prefer approved `/projects` storage for the environment, managed Python, and uv cache; if allocation
access is unavailable, ask the instructor/RC for a supported interim location. Do not delete/move
existing files or reinstall automatically.

After storage approval and review of the existing lockfile:

1. Allocate a `short` compute node with `srun` or a bounded `sbatch` job; do not install from the
   login shell. Check actual quota there using `check-quota` before large downloads.
2. Configure the approved environment/cache/managed-Python paths and record the exact commands
   here and paths in `MANIFEST.md`. A project checkout under the approved allocation can hold
   `.venv`; external environments require updating the validation script's activation path.
3. From that checkout on the compute node, use `~/.local/bin/uv sync --locked` and activate the
   approved environment. Never run concurrent installers. This is a procedure, not permission
   to start installing before the storage decision.

Python 3.12 is managed by uv. PyTorch 2.3.1 and torchvision 0.18.1 use the explicit official CUDA
12.1 index. NumPy stays on 1.26.4; OpenCV uses headless wheels. TensorBoard uses a newer compatible
range rather than the old Conda 2.6 pin. No training-method changes are intended.
`uv.lock` was retrieved from Explorer on 2026-10-22 without modifying the remote copy.
Its SHA-256 is `a9d18d4df5ab965229e9205e488b8b9938a773f3326db7c4bf281ad29dbfac4a`;
local and remote `pyproject.toml` checksums also match. Retain the lockfile in Git and use
`uv sync --locked` for reproduction after storage approval.
Check for active installers before syncing; never run concurrent syncs.
The parent `../.venv` is a separate macOS coursework environment, not this runtime.

## GPU validation

From the repository root, after sync succeeds:

```bash
sbatch tools/check_environment.sbatch
```

The 10-minute, one-GPU job loads `cuda/12.1.1`, builds the legacy NMS extensions directly (so compiler
errors are not masked by Makefile cleanup), checks entry-point imports, and tests CUDA matmul and
CPU/GPU NMS. Logs go to `slurm-logs/env-<JOBID>.out`. No training or data downloads are involved.
This script has not yet been run; compatibility is unverified. Native extension builds may modify
upstream-tracked generated C/C++/binary files; inspect diffs before committing and do not publish
machine-specific build outputs.

## Current setup status

During the latest read-only Explorer inspection (2026-10-22 review), `uv.lock` and `.venv/`
existed, `log/uv-sync.log` listed installed dependencies (including PyTorch 2.3.1+cu121), and no
`uv sync` process or queued jobs was observed. This supersedes the old 'sync still running' note.
The environment has not been exercised; directory/log presence is not proof of a usable runtime.
No GPU validation result is recorded. No installation or job was started during this local review.

Before proceeding on Explorer:

0. Agree on storage with Jack; confirm approved `/projects` access or instructor/RC guidance for
   an interim location. Inspect current quota/permissions before installing or moving anything.
1. Inspect remote diffs and preserve/review `uv.lock`; reconcile local setup changes without
   overwriting remote work. Do not launch a fresh installation just because earlier docs said pending.
2. Verify the environment interpreter/dependencies and SLURM account/QOS/GPU availability.
3. Ensure `slurm-logs/` exists, submit the bounded GPU check, and retain its job ID/output.
4. Resolve any extension/GPU compatibility failures before data-backed evaluation or training.

See `SKILL.md` for training-specific initialization, adaptation, and resume caveats.
