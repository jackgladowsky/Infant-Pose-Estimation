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

1. Reproduce the published FiDIP baseline on the SyRIP validation set.
2. Run controlled experiments intended to improve validation AP.
3. Record every experiment in `AGENTS.md` and every produced artifact in `MANIFEST.md`.
   `AGENT.md` is a compatibility symlink to `AGENTS.md` for the course submission requirement.

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
- Effective read/write/traverse access verified for `/projects/aclab` on 2026-10-22. Confirm
  this is the instructor-approved allocation and agree on a per-user path before creating anything.
- Existing checkout-local `data/`, `models/`, and `output/` directories are in personal home;
  bulk downloads/installs are paused pending agreement on approved storage (see policy below)
- GPU partition: `gpu` observed during setup; account/QOS and current GPU availability still need verification
- Dataset and pretrained weights: not yet present
- Latest read-only inspection (2026-10-22 review): remote `uv.lock` exists and installation log
  lists installed packages; no installer or queued jobs observed. GPU validation remains pending.
- No recorded completed experiment runs

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
3. Prefer an approved `/projects` allocation for `.venv`, managed Python downloads, package caches,
   datasets, and checkpoints. Confirm access and agree on exact paths before installing/downloading.
4. Prior approval for interim home storage is not blanket approval for more downloads. Following
   Jack's concern, pause new installs, bulk downloads, and moves until storage is agreed. If project
   access is unavailable, ask the instructor/RC for a supported interim location; do not invent one.
5. Inspect existing paths, ownership/permissions, quota, and active work before changes. Do not
   delete/move the current environment or caches automatically. Recreate virtual environments from
   the lockfile at an approved new path rather than assuming a moved environment remains valid.
6. Use `check-quota` and CPU-intensive disk-usage scans from a `short` compute allocation, as the
   home-quota guide directs. Filesystem-wide `df` is not a personal quota check.

## Filesystem layout

Storage plan (actual allocation still unconfirmed):

- `/projects`: preferred persistent storage for environments/caches, datasets, checkpoints, and
  retained results/logs. `/projects/aclab` is writable by this account; instructor approval and
  the exact per-user path remain unconfirmed. Access alone is not authorization.
- `/home/$USER`: personal account storage, not shared system installation space. Earlier notes
  recorded a 75 GB quota; verify the actual quota with `check-quota`. Keep code/small tools here;
  further bulky use is paused pending storage agreement.
- `/scratch`: temporary working storage for running jobs only. Persist required outputs to the
  approved `/projects` directory before a job ends; do not rely on scratch between chained jobs.

| Purpose | Path | Status |
|---|---|---|
| Local code | `/Users/jackg/neu/eece7373/Infant-Pose-Estimation` | Present |
| Explorer code | `/home/gladowsky.j/Infant-Pose-Estimation` | Cloned from fork via HTTPS |
| SyRIP data | `data/syrip/` | Parent `data/` created; dataset missing |
| Downloaded weights | `models/` | Empty directory created |
| Training outputs | `output/` | Empty directory created |
| TensorBoard/log output | `log/` | Empty directory created |
| SLURM logs | `slurm-logs/` | Empty directory created; gitignored |

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
`pyproject.toml` defines a Linux/CUDA environment and `.venv/` is gitignored. The remote install
log lists installed packages and `uv.lock` exists, but environment usability is not yet verified.
The remote lockfile has been retrieved locally with matching SHA-256; preserve both copies and
inspect changes before syncing. Check for active installers
before starting another installation. Do not use the parent coursework environment as FiDIP's runtime.

Before any further sync/install, agree on storage, preserve/review `uv.lock`, and use an allocated
compute node. Configure the virtual environment, uv cache, and managed-Python storage explicitly
for the approved location; record the exact paths/commands in `ENVIRONMENT.md` and `MANIFEST.md`.
The existing validation script assumes a repository-local `.venv`; review/update that assumption
if the chosen layout changes. Do not move the current environment blindly.

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

## Training

Command inside an allocated GPU job:

```bash
python tools/train_adaptive_model_hrnet.py \
  --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
```

A training/evaluation `sbatch` script has not been finalized: account/QOS, currently available
GPU types, and measured training resource requirements still need verification. The separate
`tools/check_environment.sbatch` is only a 10-minute environment check, not a training job.
Add tested training/evaluation submission commands here after a short pilot.

### Pre-training audit findings

Do not treat these as verified training/restart behavior or silently change them during baseline evaluation:

- `lib/nms/setup_linux.py` hard-codes `-arch=sm_61`. Validate compilation and execution on the
  allocated GPU; the course's H200 examples do not establish compatibility of this legacy extension.
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
- Use resumable, checkpointed stages and `sbatch --dependency=afterok:<JOB_ID>` to chain successful
  jobs. Checkpoints needed by later stages must live on persistent storage.
- For Bayesian optimization, an optimizer must read completed trial metrics and propose the next
  hyperparameters. SLURM dependencies only sequence jobs; they do not perform optimization.
- A future workflow can chain training → evaluation → optimizer/proposal → next trial. Keep the
  optimizer state and trial history on `/projects`, with a bounded trial/compute budget.
- Scripts and resume behavior must be verified before launching any chain; no jobs submitted yet.

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

## Active runs

| Run ID | Purpose | Config/commit | SLURM job | Status | Key artifact |
|---|---|---|---|---|---|
| — | No runs yet | — | — | — | — |

## Guardrails

- Never expose or commit secrets.
- Never commit SyRIP data or model checkpoints.
- Never run installs, environment builds, native compilation, or compute jobs on login nodes.
- Agree on approved storage before further installations, bulk downloads, or environment moves.
- Keep installs user-local; do not modify system/shared software or use `sudo`.
- Confirm before canceling a job or deleting an artifact.
- Use `git pull --ff-only` on Explorer.
- Change one primary experimental variable at a time.
- Preserve failed-run logs and document the cause in `AGENTS.md`.
- Update `MANIFEST.md` whenever data, a config, checkpoint, or result is created or moved.
