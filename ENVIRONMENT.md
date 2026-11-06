# uv environment on Explorer

This is a Linux x86-64 CUDA environment, not a macOS environment. Original Conda recipe is retained
for reference; `pyproject.toml` defines the smaller uv-managed training/evaluation environment.
Synthetic-generation dependencies are separate and not installed here.

**Source release:** this release includes the NMS/build compatibility fixes, safe/strict
checkpoint loading, validation/SLURM helpers, tests and recorded metric JSONs used for this study.
It supersedes the earlier documentation-only publication. Datasets, weights, raw prediction files
and full reference audits remain external; see the reproduction guide for command prerequisites.
The local/offline checks passed, but no new clean-install Linux/CUDA end-to-end run is claimed.

[reproducibility/README.md](reproducibility/README.md) describes the working source, input
acquisition and evaluation commands. Recorded metrics/protocols are in `reproducibility/results/`;
original datasets, weights and full reference-audit outputs remain external.

## Setup

uv is installed at `/home/gladowsky.j/.local/bin/uv` (0.12.17 at setup time).
This is a personal-account executable, not a system-wide install. It was installed without modifying
shell startup files. No uv-specific institutional endorsement was found; see `SKILL.md` for official
RC policy links and the distinction between user-local tools and large research environments.

**Jack confirmed personal home storage, with a 75 GB limit.** Keep the existing checkout at
`/home/gladowsky.j/Infant-Pose-Estimation` and its `.venv`. Managed Python is currently referenced
under `~/.local/share/uv/python`; inspect uv's actual cache path rather than assuming its location.
No `/projects` allocation is approved, and `/projects/aclab` is explicitly excluded. Do not create
an allocation, migrate files, delete caches, or reinstall automatically.

The coding agent runs on Jack's Mac/jacks-server and uses passwordless SSH to submit/monitor jobs.
Prefer bounded `sbatch` jobs for routine setup; interactive `srun` is only needed for debugging.

CPU quota/dependency inspection, native CPU/V100 checks, and pretrained Test100 evaluation
already passed in the working Explorer checkout; no reinstall or new validation run is needed.
Use the existing `.venv`. If a future dependency repair is necessary, run `uv sync --locked` in a
separate bounded CPU job with sufficient home quota; never run concurrent installers.

Python 3.12 is managed by uv. PyTorch 2.3.1 and torchvision 0.18.1 use the explicit official CUDA
12.1 index. NumPy stays on 1.26.4; OpenCV uses headless wheels. TensorBoard uses a newer compatible
range rather than the old Conda 2.6 pin. No training-method changes are intended.
`uv.lock` was retrieved from Explorer on 2026-10-22 without modifying the remote copy.
Its SHA-256 is `a9d18d4df5ab965229e9205e488b8b9938a773f3326db7c4bf281ad29dbfac4a`;
local and remote `pyproject.toml` checksums also match. Retain the lockfile in Git and use
`uv sync --locked` for reproduction when needed, in a compute allocation after the quota check.
Check for active installers before syncing; never run concurrent syncs.
The parent `../.venv` is a separate macOS coursework environment, not this runtime.

## Timed native/GPU validation (passed on V100)

The original job `11305949` ended TIMEOUT, reported elapsed 10m30s, with a CPU-NMS integer-dtype
error after native compilation, CLI imports, and CUDA matmul. Its generic GPU request wrongly
allocated H200. The revised workflow separates compilation from GPU time. Reviewed source/scripts
were synced with matching checksums. CPU job `11308157` passed all 10 harness tests but failed
in native compilation (3s) while Cython opened generated `cpu_nms.c`. Installed Cython supports
`--cython-c-in-temp`; added that flag and an explicit source-header include path. CPU retest
`11308338` produced PASS: 10 harness tests, 6 compiled CPU-NMS cases, native build 12s, script
elapsed 15s. Accounting confirms COMPLETED `0:0`, allocation elapsed 4m56s; the controller was
still cleaning up afterward. Dependent GPU check `11308377` subsequently COMPLETED `0:0`,
allocation elapsed 2m58s, and is no longer queued. All checks passed on `d1007`, Tesla
V100-SXM2-32GB (32768 MiB advertised; Torch reported 31.7 GiB), capability 7.0, driver 545.23.08.

GPU stage timings: bootstrap/inventory/hashes <1s each, combined CUDA/NMS/project checks 17s,
training CLI help 5s, evaluation CLI help 4s; total script 26s. Python substeps: CUDA/import and
matmul 8.402s, CPU NMS 0.042s, GPU NMS 0.008s, torchvision import 5.644s, dataset import 0.500s,
models import 0.028s, TensorBoard import 0.890s. All six CPU and six GPU NMS fixtures passed.
These are smoke-test timings, not model-training benchmarks. Data validation and pretrained
Test100 evaluation subsequently passed; see the baseline section below.

After reviewing and safely synchronizing the source/scripts, run from the Explorer home checkout:

```bash
build_job=$(sbatch --parsable tools/build_environment.sbatch)
sbatch --dependency=afterok:"$build_job" tools/check_environment.sbatch
```

- **CPU build:** `short`, `ib`, 2 CPUs, 4 GB host RAM, 5-minute limit. Loads `cuda/12.1.1`,
  records source hashes, tests the harness, rebuilds extensions with generated C under `build/temp`,
  and tests CPU NMS. No GPU is requested.
- **GPU check:** explicitly `gpu:v100-sxm2:1`, 2 CPUs, 4 GB host RAM, 5-minute limit. Reports
  GPU model/VRAM/driver, tests CUDA and CPU/GPU NMS, imports project modules, and checks CLI help.
  It does not compile, train, install dependencies, or download data.
- Logs: `slurm-logs/build-<JOBID>.out` and `slurm-logs/env-<JOBID>.out`. Keep both job IDs.
- `tools/environment_check_common.sh` logs UTC start/end, exit status, and elapsed seconds for
  every shell stage and the total job. Each external stage has a timeout. Bootstrap/module loading
  is timed and bounded by the overall SLURM cap.
- Python emits flushed, high-resolution timings for each check and NMS case, continues independent
  checks after a failure, returns nonzero on any failure, and dumps stacks every 60s during stalls.
  A shell timeout returns 124; it is a failed check, not permission to increase walltime blindly.
- CPU and GPU NMS regressions cover empty/single/duplicate/disjoint/mixed inputs and a 129-box case
  crossing CUDA's 64-box bitmask blocks. GPU work is synchronized before reporting timings.

The local fixes use matching `np.intp_t`/`np.intp` CPU indices, explicit threshold types, and an
empty-input guard in GPU NMS. The build targets V100 (`sm_70`) plus forward-compatible PTX;
`NMS_CUDA_ARCH=80` is an explicit A100 build option. Any A100 or V100 variant is approved:
use a typed `--gres=gpu:a100:1` or `--gres=gpu:v100-pcie:1` command-line override when appropriate.
Record the actual GPU; no extra permission is needed between these approved types. As of
the current 2026-11-01 policy, Jack also conditionally approves H100/H200 for training/fine-tuning if readily available
without a long queue; otherwise use A100/V100. This does not change the small smoke-check defaults.
Live eligible Hopper capacity and training compatibility have not yet been verified; measure in a
short pilot before extrapolating runtime or changing batch sizes.
Inspect generated native-file diffs before synchronization and never publish generated binaries.

V100 passed these checks and the later pose-only batch-8 pilot/fine-tunes. The pilot used about
4.3 GiB peak allocated GPU memory; this does not establish memory fit for the upstream full
adversarial trainer or arbitrary batch sizes.

## Current setup status

During the latest read-only Explorer inspection (2026-10-22 review), `uv.lock` and `.venv/`
existed, `log/uv-sync.log` listed installed dependencies (including PyTorch 2.3.1+cu121), and no
`uv sync` process or queued jobs was observed. This supersedes the old 'sync still running' note.
Subsequently, authorized CPU inspection job `11305792` on `d0023` produced a final PASS:

- Home quota: 0.01 TiB used / 0.07 TiB soft and hard limit (tool-rounded values); 33,220 inodes
  used / 1,000,000 soft / 1,500,000 hard. `check-quota "$HOME"` displayed backend path `/discovery/home`.
- uv 0.12.17; cache `/home/gladowsky.j/.cache/uv`.
- Python 3.12.14; resolved executable
  `/home/gladowsky.j/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12`.
- PyTorch 2.3.1+cu121, torchvision 0.18.1+cu121, NumPy 1.26.4; imports and tiny CPU matmul passed.
- `uv pip check --python .venv/bin/python` checked 51 packages: all compatible.

No sync/reinstallation was needed or performed. Output: `slurm-logs/inspect-11305792.out`.
Accounting now reports COMPLETED, exit `0:0`, elapsed 4m44s for the CPU inspection; the controller
was still in cleanup (COMPLETING) on the next inspection. GPU job `11305949` was initially submitted
with `--dependency=afterok:11305792`; its unnecessary dependency was removed using
`scontrol update JobId=11305949 Dependency=` after the successful CPU results, because the tests
are independent and cleanup was delaying eligibility. Neither job was canceled.
GPU log: `slurm-logs/env-11305949.out`; the observed CPU-NMS dtype issue is fixed in local
source, compiled on Explorer, and passed all CPU-NMS fixtures in retest `11308338`. GPU check
`11308377` also passed all CUDA/GPU-NMS/import/CLI checks. The revised workflow is validated
on V100; actual model evaluation/training and A100-specific validation are separate. Keep the failed-run log; do not reinstall uv
or change dependency pins blindly. Latest accounting reports TIMEOUT with exit `1:0`.

For future environment changes or revalidation (not required again for the completed baseline):

0. Use the confirmed home layout. Inspect current quota/permissions before installing anything;
   do not use lab storage or create an allocation if home space is insufficient.
1. Inspect remote diffs and preserve/review `uv.lock`; reconcile local setup changes without
   overwriting remote work. Do not launch a fresh installation just because earlier docs said pending.
2. Verify the environment interpreter/dependencies and SLURM account/QOS/GPU availability.
3. Ensure `slurm-logs/` exists; after source sync, run the CPU build then the dependent V100
   check, retaining both job IDs and outputs.
4. Resolve any extension/GPU compatibility failures before data-backed evaluation or training.

## Pretrained Test100 baseline reproduced (2026-10-28)

Original user-supplied HRNet file (255,354,649 bytes) was moved into `models/hrnet_fidip.pth`
and copied through xfer. SHA-256:
`bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07`.
CPU admission `11450222` verified safe weights-only load, all 1,754 keys/shapes, finite tensors,
and a CPU forward. Evaluation entry point now uses `weights_only=True`, `map_location='cpu'`,
and strict state loading; no dependency pin changes or unsafe pickle fallback.

Dependent V100-SXM2-32GB job `11450265` COMPLETED `0:0`: AP **93.6071706887983%** (published
93.6 rounded), AP50/AP75 **98.52475247524751%**, AR **94.6%**, AR50/AR75 **99%**. Original
Test100, GT boxes, flip/postprocess, official input/batch sizes preserved; no training or data repair.
GPU allocation 71s, script 64s. Full-precision metrics and provenance live at
`output/baseline-test100-11450265/metrics.json`; exact commands/artifacts are in `MANIFEST.md`.

Workflow: `tools/check_checkpoint.sbatch` on CPU, then `tools/evaluate_baseline.sbatch` with an
afterok dependency and `FIDIP_CHECKPOINT_JOB` exported. Both have 5-minute caps, no installs,
resource/model/data admission checks, and timed stages. File copies always use xfer; SLURM
submission/monitoring uses the Explorer login alias. The runtime fixes and helpers are included
in this source release; the original model/data and private Explorer working checkout are unchanged.

## Diagnostic training/resume pilot passed (2026-11-01)

CPU audit `11553238` and GPU pilot `11553500` COMPLETED `0:0`. Pilot used V100-SXM2-32GB on
d1019 (gpu-short, 4 CPUs, 8G host RAM, 5m cap); began ~34s after submission. H200 forecasts were
long, so no Hopper job submitted. Source checkpoint and original datasets remained unchanged.

Fifty supervised pose updates on 32 explicitly verified training examples had finite loss/gradients
and changed weights. Saved/restored pose+Adam+RNG reproduced 3 uninterrupted steps exactly.
Median steady step 0.38955s at batch 8, FP32/deterministic/TF32 off; peak GPU allocated 4.299 GiB,
reserved 4.990 GiB; host MaxRSS ~1.424 GiB. Script 56s, allocation accounting 135s. Compute-only
extrapolation is ~3.25m/500 steps or ~6.49m/1,000, not a full-data end-to-end runtime guarantee.
Reports: `output/finetune-pilot-11553500/{protocol.json,pilot-report.json}` on both Mac/Explorer;
~729 MiB diagnostic resume checkpoint stays Explorer home only. No Test100 scoring or AP improvement
claim. The exclusion profile was subsequently approved for the completed screen below. The pilot
and experiment helpers are included in this source release.

## Three prespecified tests completed (2026-11-01)

GPU `11553889` COMPLETED `0:0`, V100-SXM2-32GB/d1022, 4CPU/8G, 6m45s allocation / 6m18s
script. Original flip-on control exactly matched 93.60717 AP. With 600 updates, batch 8, Adam 1e-5,
seed 7373, head-only reached 93.54721 AP and stage4+head 93.19864 AP; frozen parameters/BN stats
verified unchanged. Original-weight flip-off scored 91.91587 AP; paired warmed CNN-forward timing
roughly halved (13.00666 to 6.50483 ms/image), NOT a 2x end-to-end application-speed claim.

Both fine-tunes used the user-approved 1,150-record selection manifest (196 real/954 synthetic),
with no source-label/image edits; same balanced sampling and geometry schedule. No Test100 early
stopping, adaptive sweep, expanded data or new adversarial classifier. Single-seed exploratory
results do not establish statistical significance; original weights remain best of these tests.

CPU prep `11553804` passed seven Linux unit tests/data validation and exited rc0/18s, but allocation
later reported TIMEOUT0:0/5m08s (batch later COMPLETED0:0). Lifecycle/accounting discrepancy retained,
cause unverified. GPU independently verified the approved manifest and pinned input hashes.

Results: `output/small-screen-11553889/{results.json,protocol.json,summary.md}`; predictions/protocols
available on Mac and Explorer via xfer. Checkpoints retained in Explorer home only. No active jobs,
new dependency installs or original-checkpoint changes. At run completion the source was still
unpublished; its separate source release was authorized later.

## Bounded inference follow-ups completed (2026-11-02 UTC)

CPU analysis `11556695` COMPLETED0:0/17s; nine geometry/BN tests passed on Linux. Existing
predictions and buffers analyzed without new inference;131 BN modules had393 changed buffers.
GPU `11556720` COMPLETED0:0 on V100-SXM2-32GB/d1022,2CPU/8G,47s allocation/45s script,
within one10m cap. No new training or repeated trials; original/trained control AP reproduced exactly.

Restoring only original BN running buffers in the stage4-trained model reached93.35239 AP
(+0.15376 vs trained, still -0.25478 vs original). Fixed0.9/1/1.1 crop TTA on original weights
reached93.32060 AP (-0.28657), with triple forward-image views. Original93.60717 remains best.
Crop fusion follows repo affine conventions and coverage normalization; no scale sweep.

Reports/protocols/predictions/summary: `output/bounded-diagnostics-11556720/` on Mac and Explorer.
Large diagnostic checkpoint stays Explorer home. Original data/weights unchanged, no active jobs,
new installs, code pushes or further training. Stop at the agreed budget gate.

See `SKILL.md` for training-specific initialization, adaptation, and resume caveats.
