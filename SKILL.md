---
name: fidip-explorer
summary: Operate the FiDIP project locally and on Northeastern's Explorer SLURM cluster.
---

# FiDIP Project Operations

Read this file before changing the environment, moving data or checkpoints, or operating Explorer.
Keep commands and the active-runs table current as the project evolves.

The course's general Explorer guide is also included at
[`docs/explorer/SKILL.md`](docs/explorer/SKILL.md). It is an unchanged reference template;
this root guide supplies this project's actual connection facts, paths, and guardrails.
Its notebook/GPU-recipe references belong to the original course repo, not `docs/explorer/`.
SSH access already works: do not rerun its key-generation example or overwrite existing keys.

## Project objective

1. Reproduce the published FiDIP baseline on original SyRIP Test100.
2. Run controlled experiments intended to improve validation AP.
3. Record every experiment in `AGENTS.md` and every produced artifact in `MANIFEST.md`.
   `AGENT.md` is now a short, ordinary Markdown course entry point linking to the canonical notebook.

Run instructions are in [reproducibility/README.md](reproducibility/README.md), with recorded
metrics/protocols under `reproducibility/results/`. Additional model runs and Git commit/push
still require approval; datasets, weights and full audit logs stay outside Git.

Published HRNet-W48/DarkPose reference results from `README.md`:

- Fine-tune: **92.7 AP**
- FiDIP: **93.6 AP**

## Current status

- Local checkout: `/Users/jackg/neu/eece7373/Infant-Pose-Estimation`
- Git `origin`: Jack's GitHub fork
- Explorer username: `gladowsky.j`
- Explorer SSH alias: configured in `~/.ssh/config`; non-interactive key-based access verified (`hostname`: `explorer-01`, user: `gladowsky.j`)
- Explorer code path: `/home/gladowsky.j/Infant-Pose-Estimation`; setup synchronized from the
  fork at `8f4c497` on 2026-10-22 (check Git history for later documentation updates).
  Original remote setup is preserved in a named Git stash; do not drop it automatically.
- Jack confirmed personal home `/home/gladowsky.j` for this project, with a 75 GB limit.
  Keep the existing home checkout and uv environment; inspect quota before further downloads.
- Do not use `/projects/aclab` or other lab/group allocations. Do not create/request a new project
  allocation without Jack's approval; writable access is not authorization.
- Scored allocations used typed V100-SXM2 on `gpu` / `gpu-short`, default account `ostadabbas`,
  QOS `normal`. These are historical facts; check current eligibility/capacity before any future job.
- Dataset prepared locally on 2026-10-28 under gitignored `data/`: retained ZIPs in `archives/`,
  untouched trees in `raw/`, working relative-symlink profile in `syrip/`. All 7,657 images decode.
  Original Test100 (100 samples) is baseline-ready; FiDIP training has 46 dimension mismatches
  and 4 byte-identical Test100 overlaps, so do not call training clean/ready without review.
  See `data/README.md`, `data/validation-report.json`, and `MANIFEST.md`. The complete data tree
  was copied to Explorer home through `xfer.discovery.neu.edu` on 2026-10-28 (rsync exit 0);
  CPU validation `11449533` COMPLETED `0:0`: both archive hashes, all 7,657 image hashes/
  dimensions, and all source/working annotation hashes match the Mac. Test100 is ready on Explorer;
  training caveats persist. Original HRNet weights are present and CPU-verified; baseline GPU
  job `11450265` reproduced 93.6 AP (93.60717 unrounded) with COMPLETED `0:0`. Diagnostic
  pilot `11553500` and three-test screen `11553889` passed. Head-only93.54721 / stage4+head93.19864 /
  original flip-off91.91587 AP. Bounded BN/TTA diagnostics `11556720` also completed below baseline
  (93.35239/93.32060 AP) using47s GPU allocation. Original remains best; no active jobs or more runs authorized.
- CPU inspection job `11305792` produced PASS for existing Python 3.12.14, torch/torchvision/numpy
  imports, tiny CPU matmul, and consistency of all 51 installed packages. No reinstall needed.
- Home quota tool reported 0.01 TiB used / 0.07 TiB soft and hard limit (rounded), 33,220 inodes
  used / 1,000,000 soft / 1,500,000 hard. uv cache: `/home/gladowsky.j/.cache/uv`.
- CPU log: `slurm-logs/inspect-11305792.out`; accounting reports COMPLETED, exit `0:0`,
  elapsed 4m44s. Controller cleanup was still COMPLETING on the subsequent check.
- GPU/native validation job `11305949` used an unconstrained GPU request and was assigned an
  H200 on `d4055`; this was a resource-selection mistake. Final state TIMEOUT, elapsed 10m30s.
  Native builds, train/eval entry-point imports, and CUDA matmul passed; CPU NMS failed with
  `ValueError: Buffer dtype mismatch, expected 'int_t' but got 'int'`. GPU NMS was not reached.
  An earlier Python file-I/O wait was observed; the final log contains the concrete dtype failure.
  CPU job `11308157` passed 10 harness tests, then failed generating C in the source directory.
  Revised CPU retest `11308338` uses `--cython-c-in-temp` plus a source-header include path:
  all 10 harness tests and all 6 compiled CPU-NMS cases PASS; native build 12s, script total 15s.
  Accounting confirms COMPLETED `0:0`, allocation elapsed 4m56s (distinct from 15s script timing).
  Typed-V100 check `11308377` is COMPLETED `0:0`, out of the queue: all CUDA/NMS/import/CLI
  checks PASS on V100-SXM2-32GB (`d1007`, driver 545.23.08). Script 26s; allocation 2m58s.
  No active validation/baseline jobs remain; Test100 baseline reproduction also passed as
  `11450265`. The later approved pilot, three-test screen and two diagnostics are complete.
  Next is final submission review, not more training.
- Jack permits any A100 or V100 variant. Current policy, 2026-11-01: H100/H200 are conditionally approved
  for training/fine-tuning when readily available without a long queue; otherwise use A100/V100.
  Check live eligible capacity, use typed GRES, and record the allocation. Hopper training support
  still needs a pilot; this permission does not change the economical smoke-check defaults.

## Runtime

The checked-in `fidip_env.yml` specifies:

- Conda environment: `fidip`
- Python 3.12.4
- PyTorch 2.3.1
- CUDA runtime 12.1

Training requires Linux and an NVIDIA GPU. Use the Mac checkout for code review and lightweight
analysis; use Explorer for native library compilation, evaluation, and training. Do not build the
Linux NMS extension on macOS.

## Secrets

- Keep credentials only in `.env`; it is gitignored.
- `.env.example` documents allowed variable names without values.
- Never print, log, paste, or commit `.env` contents.
- Load variables without echoing them, for example: `set -a; source .env; set +a`.

## Explorer connection setup

Configured locally; non-interactive authentication verified:

```sshconfig
Host explorer
    HostName login.explorer.northeastern.edu
    User gladowsky.j
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 60
```

Verify non-interactive access:

```bash
ssh -o BatchMode=yes explorer 'hostname; id -un'
```

If this requests a password or 2FA, stop and complete key installation manually before asking an
agent to operate the cluster.

The agent runs on Jack's Mac or jacks-server and operates Explorer over passwordless SSH; do not
install/run the coding agent on the cluster by default. Submit bounded `sbatch` jobs remotely and
monitor them with `squeue`/`sacct` and logs. Interactive `srun` is optional for debugging, not required
for routine setup; both allocate compute nodes.

## Required file-transfer route — use xfer, NOT the login node

**All manual file copies between Mac/jacks-server and the cluster use
`gladowsky.j@xfer.discovery.neu.edu`, not the `explorer` login alias.** This includes datasets,
archives, checkpoints, exported logs, and manual code/docs copies (`rsync`, `scp`, streamed tar).
Use `ssh explorer` for lightweight commands, Git operations, SLURM submission/status, and live log
inspection; do not use it as the endpoint for file-copy commands.

Official requirement, checked 2026-10-28:
[RC Transfer Data](https://rc-docs.northeastern.edu/en/latest/datamanagement/transferringdata.html).
RC currently documents the Discovery transfer host for Explorer too. Verified `xfer-00` exposes
the same `/home/gladowsky.j/Infant-Pose-Estimation` path as Explorer. This changes transport,
**not** Jack's choice of personal home storage. No lab/project allocation is authorized.

```bash
# Transfer access: existing key, passwordless, strict host-key verification.
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 \
  -o StrictHostKeyChecking=yes gladowsky.j@xfer.discovery.neu.edu 'hostname'

# Data transfer from Mac/server; preserve relative links, permissions, and timestamps.
# Never add --delete or overwrite conflicting remote work without review.
rsync -rltp --partial --stats \
  -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes' \
  data/ gladowsky.j@xfer.discovery.neu.edu:/home/gladowsky.j/Infant-Pose-Estimation/data/

# Commands/jobs still go through the Explorer login alias.
ssh explorer 'cd ~/Infant-Pose-Estimation && squeue -u "$USER"'
```

No xfer SSH alias has been added; use the explicit hostname above. Never bypass host-key checks,
replace existing keys, or assume transfer-host authentication works just because login SSH does.
Bulk transfers through the login node were attempted before this requirement was checked and
returned unexpected EOF/partial copies. The corrected xfer transfer completed successfully;
record the lesson, do not repeat the old route or assume the EOF cause was definitively proven.
Large-transfer alternative: RC recommends Globus. Intra-cluster large copies/hash/image audits
belong in CPU allocations, not on login nodes.

## Official policy and installation guardrails

Reviewed 2026-10-22 against Northeastern RC documentation:

- [General policies](https://rc.northeastern.edu/general-policies/): do not run jobs on login
  nodes; cluster use must be Northeastern-sponsored research/coursework.
- [Local software installation](https://rc-docs.northeastern.edu/en/latest/software/fromsource/makefile.html):
  user-local installs in writable home or project directories are supported; this is not a
  system-wide installation or permission to modify shared software.
- [Home storage guidance](https://rc-docs.northeastern.edu/en/latest/best-practices/homequota.html):
  home has fixed quotas; research environments should use `/projects`. Avoid installing Python
  packages outside an isolated environment. The `.local` package warning does not mean a small
  standalone executable in `~/.local/bin` is prohibited.
- [Environment setup guidance](https://rc-docs.northeastern.edu/en/latest/software/packagemanagers/conda.html):
  allocate a compute node before creating/installing environments. Apply this workflow to uv
  dependency installation and native compilation too (our conservative project rule).
- No uv-specific policy or explicit uv endorsement was found. uv is the project's chosen
  user-local environment manager, not a claim of RC certification. Ask RC/instructor when uncertain.

Operational rules for this project:

1. Login nodes: lightweight editing, Git/SSH inspection, and job submission/monitoring only.
   Run `uv sync`, dependency builds, native compilation, evaluation, and training in `srun`/`sbatch`
   allocations; request a GPU only for steps that need it.
2. Keep uv user-local (existing `/home/gladowsky.j/.local/bin/uv`); never use `sudo`, install into
   system directories, alter shared modules, or overwrite SSH keys. User-local does not guarantee
   privacy: filesystem permissions and administrator access still apply.
3. RC recommends `/projects` for bulky research environments, but Jack has explicitly chosen his
   personal home for this project. Keep `.venv`, managed Python/cache, and retained artifacts within
   `/home/gladowsky.j`, respecting its 75 GB limit. Do not infer permission to use any group storage.
4. Storage choice is settled, not blanket permission for arbitrary downloads. First run a bounded,
   read-only CPU batch check of quota and the existing environment. Do not reinstall unnecessarily.
   If space is insufficient, stop and ask Jack/instructor/RC; do not create a project allocation,
   switch storage, delete caches, or move environments automatically.
5. Inspect existing paths, ownership/permissions, quota, and active work before changes. Do not
   delete/move the current environment or caches automatically. Recreate virtual environments from
   the lockfile at an approved new path rather than assuming a moved environment remains valid.
6. Use `check-quota` and CPU-intensive disk-usage scans from a `short` compute allocation, as the
   home-quota guide directs. Filesystem-wide `df` is not a personal quota check.

## Filesystem layout

Storage plan confirmed by Jack:

- `/projects`: performant, persistent storage in general, but no allocation is approved for this
  project. Do not use `/projects/aclab` or create a project allocation on our own.
- `/home/$USER`: chosen persistent storage, 75 GB limit. Use the existing
  `/home/gladowsky.j/Infant-Pose-Estimation` checkout and `.venv`; quota includes tools, managed Python,
  caches, datasets, logs, and checkpoints across the entire home. Verify current usage/limits.
- `/scratch`: temporary working storage for running jobs only. Persist required outputs to the
  chosen home location before a job ends; do not rely on scratch between chained jobs.

| Purpose | Path | Status |
|---|---|---|
| Local code | `/Users/jackg/neu/eece7373/Infant-Pose-Estimation` | Present |
| Explorer code | `/home/gladowsky.j/Infant-Pose-Estimation` | Cloned from fork via HTTPS |
| SyRIP data | Local and Explorer `data/syrip/` → `data/raw/` | Transfer and Test100 validated by CPU job `11449533`; training caveats remain |
| Downloaded weights | Mac/Explorer `models/hrnet_fidip.pth` | Original HRNet acquired, SHA-256/strict CPU load verified; Test100 baseline reproduced |
| Training outputs | `output/` | Completed baseline/screen/diagnostic outputs; large trained/resume weights on Explorer |
| TensorBoard/log output | `log/` | Verified logs, analysis and local evidence copies |
| SLURM logs | `slurm-logs/` | Completed job logs on Explorer; selected copies in the reproduction bundle |

`data/`, `models/`, `output/`, and `log/` are gitignored. Record actual external paths and symlinks
in `MANIFEST.md`; never commit datasets or checkpoints.

## Repository synchronization

```bash
# Local: publish a reviewed change to the fork
git push origin master

# Explorer: update without creating an implicit merge commit
ssh explorer 'cd ~/Infant-Pose-Estimation && git pull --ff-only origin master'
```

Do not force-push shared experiment history. Initial setup files were copied from the Mac to
Explorer before publication. Avoid concurrent edits to both copies. Before subsequent syncs,
inspect remote status/diffs and preserve changes; do not blindly overwrite the remote checkout.
Explorer's HTTPS remote supports public pulls; push authentication has not been configured.

## Environment and native extension

Preferred setup is now **uv**; see `ENVIRONMENT.md` for commands and validation status.
`pyproject.toml` defines a Linux/CUDA environment and `.venv/` is gitignored. CPU and full V100 native/GPU smoke checks now pass after the NMS/build-path fixes (`11308338`,
`11308377`). `uv.lock` is preserved and no reinstall is needed. Real dataset/model evaluation
and training remain separate from those smoke checks. Pretrained Test100 evaluation subsequently
reproduced the published result (`11450265`). The pose-only pilot and two controlled 600-step
fine-tunes subsequently passed; original adversarial training from scratch was not reproduced.
The remote lockfile has been retrieved locally with matching SHA-256; preserve both copies and
inspect changes before syncing. Check for active installers
before starting another installation. Do not use the parent coursework environment as FiDIP's runtime.

Use the existing home-based layout; preserve/review `uv.lock` and inspect quota/environment in a
short CPU batch allocation before deciding whether `uv sync --locked` is needed. Record the actual
cache/managed-Python paths and any setup commands in `ENVIRONMENT.md` and `MANIFEST.md`.
The validation script assumes the existing repository-local `.venv`; no layout migration is planned.
Do not move/reinstall the current environment blindly.

`fidip_env.yml` is retained as the original Conda reference, not the active workflow. Do not run
an additional Conda installation. If uv resolution fails, preserve the error in `AGENTS.md` before
changing dependency pins.

## Baseline evaluation

Required files:

- `data/syrip/annotations/...`
- `data/syrip/images/...`
- `models/hrnet_fidip.pth`

Command inside an allocated GPU job:

```bash
python tools/test_adaptive_model.py \
  --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
  TEST.MODEL_FILE models/hrnet_fidip.pth TEST.USE_GT_BBOX True
```

Record AP/AP50/AP75/AR and the exact checkpoint/config in both `AGENTS.md` and `MANIFEST.md`.

### Reproduced pretrained baseline (2026-10-28)

Original user-supplied `models/hrnet_fidip.pth` SHA-256:
`bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07`.
CPU admission `11450222` verifies safe weights-only deserialization, exact keys/shapes, finite
weights, and a CPU forward before GPU time. Entry point now loads on CPU with `weights_only=True`
and `strict=True`; do not restore lenient/unsafe loading to conceal checkpoint mismatches.

```bash
# From the home checkout; record IDs and use typed A100/V100 GRES.
check_job=$(sbatch --parsable tools/check_checkpoint.sbatch)
sbatch --dependency=afterok:"$check_job" --export=ALL,FIDIP_CHECKPOINT_JOB="$check_job" \
  tools/evaluate_baseline.sbatch
```

Each job is capped at 5m. The GPU script defaults to V100-SXM2; an approved A100/V100 override
is allowed. It keeps original Test100/GT boxes/flip/postprocess/input size/batch size unchanged,
turns off debug rendering only, logs every batch, and saves full-precision COCO metrics.
Successful GPU run `11450265`: V100-SXM2-32GB, `d1020`, COMPLETED `0:0`, 71s allocation / 64s
script. AP **93.6071706887983%**, AP50/AP75 **98.52475247524751%**, AR **94.6%**, AR50/AR75 **99%**.
This matches published rounding, not a new improvement. Metrics: `output/baseline-test100-11450265/metrics.json`.
Inputs, all artifact paths, and source hashes are recorded in `MANIFEST.md`. Source remains
uncommitted over `7bc14bc`; no new push yet. Training caveats are not waived by baseline success.

## Training

Command inside an allocated GPU job:

```bash
python tools/train_adaptive_model_hrnet.py \
  --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
```

The command above is the upstream trainer, not the driver for our reported continuations.
Validated helpers are `tools/pilot_finetune.sbatch`, `tools/prepare_small_screen.sbatch` and
`tools/run_small_screen.sbatch`. See [reproducibility/README.md](reproducibility/README.md) for
submission/dependency commands and the matched 600-step protocol. Diagnostic jobs are
`tools/analyze_bounded_diagnostics.sbatch` and `tools/run_bounded_diagnostics.sbatch`.
They are documented for reproducibility, not permission to launch additional work.

### Diagnostic pilot verified (2026-11-01)

CPU admission `11553238` and V100 pilot `11553500` passed. Typed V100 on gpu-short started
in ~34s; H200 forecasts were long so no Hopper job submitted. The pilot performs 50 pose-only
Adam updates on 16 real + 16 synthetic verified non-overlap samples, then verifies 3 uninterrupted
versus 3 resumed steps with exact pose/Adam/RNG equivalence. Original checkpoint unchanged.
Median step 0.38955s at batch 8, FP32/TF32 off/deterministic; peak GPU allocated 4.299 GiB,
reserved 4.990 GiB, host MaxRSS ~1.424 GiB. Script 56s, allocation 135s. New diagnostic checkpoint
is ~729 MiB and includes optimizer/RNG, not a raw final pose file. No Test100 evaluation.

`output/finetune-pilot-11553500/pilot-report.json` contains exact protocol/timings. The author
training sampler is already 1:1, despite the raw 1:5 inventory. A future real-heavy comparison must
explicitly differ (e.g. 2:1). At the pilot stage the 1,150-record exclusion profile was unapproved;
Jack subsequently approved it for the completed three-test screen `11553889` (see active/history table).
Original releases were not repaired; the five-run proposal was superseded by the three-test plan.
The pose-only pilot helpers are included in the source release; they do not fix the original
adversarial trainer's resume behavior.

### Pre-training audit findings

Do not treat these as verified training/restart behavior or silently change them during baseline evaluation:

- The original NMS build hard-coded `sm_61`; the local revision targets V100 `sm_70` plus PTX
  with explicit `NMS_CUDA_ARCH` override (80 for A100). Validate on the actual allocated GPU.
- CPU NMS dtype mismatch, removed scalar aliases in `.pyx`, and GPU empty-input handling passed
  compiled CPU/GPU regressions (`11308338`, `11308377`). Five `np.float` dtype aliases remain
  in `lib/dataset/coco.py` (not the active SyRIP loader); modernize before using COCO data paths.
- The infant HRNet YAML has `TRAIN.LAMBDA: 0.000`, `TRAIN.PRE_EPOCH: 0`, and empty
  `TRAIN.CHECKPOINT`/`MODEL.PRETRAINED`. Confirm the intended pretrained initialization and
  domain-adaptation settings before calling a training run FiDIP reproduction.
- `tools/train_adaptive_model_hrnet.py` restores pose weights/optimizer but does not save or
  restore the domain classifier or its optimizer/scheduler. Verify/fix complete resume state
  before chaining training jobs; `AUTO_RESUME: true` alone is not enough.
- The YAML's default test weight path uses `×` instead of ASCII `x`. Always override
  `TEST.MODEL_FILE` explicitly for the published pretrained baseline.
- Keep the original and expanded SyRIP/checkpoint versions distinct; record checksums and splits
  so comparisons against the published Test100 reference are meaningful.

### Scheduling and optimization policy

- Request only the walltime needed, informed by a short pilot run; do not default to 24 hours.
- Any A100 or V100 is approved: explicitly request `--gres=gpu:v100-sxm2:1`,
  `--gres=gpu:v100-pcie:1`, or `--gres=gpu:a100:1`. The script's default is V100-SXM2, but a
  command-line GRES override may select another approved type without further permission. Record
  and verify the actual allocation. Never submit plain `--gres=gpu:1`. For training/fine-tuning,
  Jack also conditionally approves typed H100/H200 requests if readily available without a long
  queue (current policy confirmed 2026-11-01); use the site's verified type names after checking live capacity.
  Fall back to A100/V100 instead of waiting for Hopper. Other GPU types still require approval.
- Use resumable, checkpointed stages and `sbatch --dependency=afterok:<JOB_ID>` to chain successful
  jobs. Checkpoints needed by later stages must live on persistent storage.
- For Bayesian optimization, an optimizer must read completed trial metrics and propose the next
  hyperparameters. SLURM dependencies only sequence jobs; they do not perform optimization.
- A future workflow can chain training → evaluation → optimizer/proposal → next trial. Keep the
  optimizer state and trial history in the approved home checkout, with a bounded trial/compute
  budget and quota-aware checkpoint retention.
- Scripts and resume behavior were verified in the completed pilot and bounded screen. The agreed
  stop gate has been reached; additional training requires a new request.

## CPU quota/environment inspection

`tools/inspect_environment.sbatch` requests `short`, `ib`, 1 CPU, 2 GB RAM, and at most 5 minutes.
It runs `check-quota "$HOME"`, reports existing uv/Python/package versions, checks tiny CPU tensor
operations and dependency consistency. No sync, downloads, native builds, GPU, or disk cleanup.
Submit with `ssh explorer 'cd ~/Infant-Pose-Estimation && sbatch --parsable tools/inspect_environment.sbatch'`.
The initial authorized submission is job `11305792`; do not resubmit merely because SSH timed out.
Check `squeue` and `sacct` first. Logs: `slurm-logs/inspect-<JOBID>.out`.

## Dataset transfer validation

After an xfer-only data copy and preserving the local audit as `data/image-manifest.local.json`
and `data/validation-report.local.json`, submit from the home checkout:

```bash
ssh explorer 'cd ~/Infant-Pose-Estimation && sbatch --parsable tools/validate_dataset.sbatch'
```

This is `short`/`ib`, 4 CPUs, 4 GB RAM, 5-minute cap, no GPU/install/training. It SHA-256 checks
both ZIPs, fully decodes/hashes every image, compares annotation hashes, verifies relative working
links and Test100 readiness, and requires the same documented training caveats as the source audit.
The raw auditor intentionally exits 1 for known training issues; the wrapper exits 0 only if the
transfer is identical and Test100 passes. It does **not** waive those training problems.

Validated job `11449533`: COMPLETED `0:0`, `d0005`, allocation elapsed 31s, script 28s.
Remote evidence: `slurm-logs/data-11449533.out` and `data/explorer-transfer-validation.json`.
Local copies: `data/explorer-validation/2026-10-28/`. Copy/export these through xfer, not login.

## Timed CPU-build / approved-GPU validation workflow

See `ENVIRONMENT.md` for the revised stage limits and checks. After safe source synchronization:

```bash
# On Explorer, from the home checkout; no automatic submission during local review.
build_job=$(sbatch --parsable tools/build_environment.sbatch)
sbatch --dependency=afterok:"$build_job" tools/check_environment.sbatch
```

CPU compilation, local harness tests, and NMS regression use `short` without a GPU. The GPU
check defaults to typed V100-SXM2; an approved A100/V100 override is allowed. Each job has a
5-minute cap and stage timings/timeouts. Keep logs/job IDs and
inspect `STEP_END`/`STAGE_END` and final SLURM state; a build alone is not GPU validation.

### GPU inventory snapshot (2026-10-22)

Observed in the public `gpu` partition using `sinfo -N -p gpu -h -o '%N|%t|%G|%m|%f'`:

| GPU class | Exact GRES type | Project use |
|---|---|---|
| V100 SXM2 | `gpu:v100-sxm2:1` | Approved; script default |
| V100 PCIe | `gpu:v100-pcie:1` | Approved alternative |
| A100 | `gpu:a100:1` | Approved alternative |
| T4 | `gpu:t4:1` | Present; not the chosen validation target |
| H200 | `gpu:h200:1` | Conditionally approved for training if readily available; snapshot is not live capacity |

General deep-learning training horsepower, roughly: **H200 > A100 > V100 SXM2 > V100 PCIe > T4**.
This is workload-dependent, not a benchmark for this project; bigger is not automatically the right
resource. A100/V100 remain approved defaults. Jack now permits H100/H200 for training/fine-tuning
only when readily available without a long wait; H100 availability/type was not established by this
historical inventory. Do not infer speedups or live capacity from the ranking.

This is installed inventory, not reserved/free capacity. No node was fully idle in this snapshot;
`mix` alone does not prove a GPU is free. Some V100 nodes were draining/nonresponding. Inventory
and queue state change; query again before submission. `sinfo`'s memory column is host RAM, not GPU
VRAM; the revised GPU check reports actual VRAM. V100 suffices for a smoke test; training fit needs
a measured pilot. Do not infer pricing from model names or scheduling priority.

## SLURM monitoring

```bash
ssh explorer 'squeue -u $USER --format="%.10i %.12P %.24j %.2t %.12M %R"'
ssh explorer 'sacct -u $USER --starttime=now-2days --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS'
ssh explorer 'seff <JOB_ID>'
```

Never cancel a running job without Jack's explicit confirmation:

```bash
ssh explorer 'scancel <JOB_ID>'
```

## Run history

No FiDIP experiment remains active as of the last completed-run check. This is historical status,
not a live query of all jobs in Jack's account.

| Run ID | Purpose | Config/commit | SLURM job | Status | Key artifact |
|---|---|---|---|---|---|
| CPU inspection | Home quota and existing uv environment | `tools/inspect_environment.sbatch` (local uncommitted script, base `7bc14bc`) | `11305792` | PASS; accounting COMPLETED `0:0`; controller cleanup observed | `slurm-logs/inspect-11305792.out` |
| GPU validation | Native NMS build, imports, CUDA/NMS execution | Original untyped-GPU script, `7bc14bc` | `11305949` | TIMEOUT; builds/imports/CUDA matmul passed; CPU NMS dtype failure, GPU NMS untested | `slurm-logs/env-11305949.out` |
| Revised CPU build | Harness, native compilation, CPU NMS regressions | Uncommitted reviewed source over `7bc14bc`; hashes logged | `11308157` | Harness 10/10 PASS; native build rc=1, generated-C path error | `slurm-logs/build-11308157.out` |
| Temporary-C retest | Generate C under build/temp; harness/native CPU checks | Updated build/setup/test hashes logged | `11308338` | PASS, COMPLETED `0:0`; harness 10/10, CPU NMS 6/6; script 15s, allocation 4m56s | `slurm-logs/build-11308338.out` |
| Revised GPU validation | CUDA/NMS/project/CLI checks, V100-SXM2-32GB | Synced uncommitted scripts over `7bc14bc` | `11308377` | PASS, COMPLETED `0:0`, no longer queued; script 26s, allocation 2m58s | `slurm-logs/env-11308377.out` |
| Data quota | Fresh pre-transfer personal-home quota | Inline read-only CPU command | `11448779` | COMPLETED `0:0`, 1s; 0.01 / 0.07 TiB used/limit (rounded) | `slurm-logs/data-quota-11448779.out` |
| Dataset transfer validation | Archive/image/annotation hashes and Test100, no GPU | `tools/validate_dataset.sbatch`; hashes logged | `11449533` | PASS, COMPLETED `0:0`; script 28s, allocation 31s; training caveats retained | `slurm-logs/data-11449533.out` |
| CPU checkpoint admission | SHA-256, safe strict load, finite tensors, CPU forward | `tools/check_checkpoint.sbatch`; original HRNet | `11450222` | PASS, COMPLETED `0:0`, 13s allocation | `log/checkpoint-11450222.json` |
| Published baseline reproduction | Original Test100, GT boxes, flip/postprocess | `tools/evaluate_baseline.sbatch`; hashes in manifest/log | `11450265` | PASS, COMPLETED `0:0`; 93.60717 AP; V100, no training | `output/baseline-test100-11450265/metrics.json` |
| Fine-tuning CPU audit | Payload checks, data caveats, resume unit tests | `tools/audit_finetune.sbatch`; diagnostic-only admission | `11553238` | PASS, COMPLETED `0:0`; 25s script; 1,150 unflagged candidates NOT approved as experiment split | `output/finetune-readiness-11553238.json` |
| Diagnostic training/resume pilot | 32 verified samples only, 50 updates + resume branches, no Test100 | `tools/pilot_finetune.sbatch`; gpu-short, typed V100, 8G override/4CPU/5m | `11553500` | PASS, COMPLETED `0:0`, out of queue; 0.38955s/step, peak GPU 4.299 GiB; exact model/Adam/RNG resume | `output/finetune-pilot-11553500/pilot-report.json` |
| Three-test CPU preparation | User-approved exclusion selection, no label edits, screen/resume unit tests | `tools/prepare_small_screen.sbatch`; short/ib/2CPU/4G/5m | `11553804` | Script PASS rc0, 18s; allocation TIMEOUT 0:0, 5m08s, lifecycle discrepancy retained | `slurm-logs/screen-prep-11553804.out` |
| Three prespecified tests | Head-only / stage4+head 600 steps; original-weight flip-off | `tools/run_small_screen.sbatch`; gpu-short, typed V100, 4CPU/8G/15m | `11553889` | COMPLETED `0:0`, 6m45s allocation / 6m18s script, out of queue; AP93.54721 / 93.19864 / 91.91587; original93.60717 remains best | `output/small-screen-11553889/results.json` |
| Existing-artifact CPU analysis | Saved prediction OKS and BN drift; nine geometry/BN tests | `tools/analyze_bounded_diagnostics.sbatch`; short/ib/2CPU/4G/5m | `11556695` | COMPLETED `0:0`,17s; no model inference/training | `output/bounded-analysis-11556695.json` |
| Two bounded inference diagnostics | Original BN buffers restored; fixed0.9/1/1.1 crop ensemble | `tools/run_bounded_diagnostics.sbatch`; gpu-short/typedV100/2CPU/8G/10m cap | `11556720` | COMPLETED `0:0`,47s allocation/45s script; AP93.35239 /93.32060; both below original; STOP | `output/bounded-diagnostics-11556720/results.json` |

## Guardrails

- Never expose or commit secrets.
- Never commit SyRIP data or model checkpoints.
- Never run installs, environment builds, native compilation, or compute jobs on login nodes.
- File copies MUST use `xfer.discovery.neu.edu`; `explorer` is for commands/SLURM, not rsync/scp/tar endpoints.
- Use only Jack's chosen home storage; check its 75 GB quota before downloads and ask before
  changing storage, moving environments, or creating/requesting project allocations.
- Keep installs user-local; do not modify system/shared software or use `sudo`.
- Confirm before canceling a job or deleting an artifact.
- Use `git pull --ff-only` on Explorer.
- Change one primary experimental variable at a time.
- Preserve failed-run logs and document the cause in `AGENTS.md`.
- Update `MANIFEST.md` whenever data, a config, checkpoint, or result is created or moved.
