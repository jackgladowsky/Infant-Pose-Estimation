# FiDIP Project Context and Agent Lab Notebook

## Start here

- Objective: reproduce the published HRNet-W48 FiDIP baseline on SyRIP Test100 (**93.6 AP**),
  then run controlled improvement experiments. Published scores are not reproduced results.
- Read `SKILL.md` before operating Explorer, changing the environment, or moving artifacts.
  Read `ENVIRONMENT.md` for uv setup and validation, and `MANIFEST.md` for inputs/results.
  The generic course Explorer guide is vendored at `docs/explorer/SKILL.md`; the root
  `SKILL.md` remains authoritative for this project's actual paths and safety rules.
- Use this Mac checkout for edits and lightweight checks. Use Explorer SLURM GPU allocations
  for CUDA builds, evaluation, and training; never run heavy work on a login node.
  Dependency installation/environment creation must also run in compute allocations. Further
  installs, bulk downloads, and moves are paused until Jack agrees on approved storage paths;
  prefer `/projects` for large environments/caches and ask RC/instructor if access is unavailable.
- Do not confuse the parent `eece7373/.venv` notebook environment with this project's separate
  Linux/CUDA environment. Do not run this project's `uv sync` on macOS.
- Keep secrets in gitignored `.env` files; never expose their values. Keep datasets and weights
  outside Git. Confirm before canceling jobs or deleting artifacts.
- Inspect both local and remote diffs before synchronization; preserve/review `uv.lock`
  (retrieved locally from Explorer with matching checksum on 2026-10-22).
  Do not blindly copy over the Explorer checkout or commit/push without user approval.
- Record hypotheses, exact commands, configs, code revisions, inputs, run IDs, metrics, failures,
  and conclusions here as work happens. Update `MANIFEST.md` for produced/acquired artifacts
  and the active-runs table in `SKILL.md` for jobs.

## Course requirements and nearby references

The course requires baseline reproduction, controlled attempts to improve accuracy (including
failed attempts), and a final report covering baseline results, experiments, best configuration,
and reflection on the agentic workflow. Root deliverables are `SKILL.md`, `AGENT.md`, and
`MANIFEST.md`. `AGENTS.md` is the canonical context/notebook; `AGENT.md` is a compatibility
symlink to it, preserving the rubric's filename without maintaining duplicate notebooks.

Local course references (outside this Git repository; may not exist on Explorer):

- `../STUDENT_GUIDE.md`: FiDIP assignment, research workflow, and final submission requirements.
- The generic Explorer operations template from
  `../EECE-7373-Machine-Learning-with-Small-Data/SKILL.md` is now copied into this repo at
  `docs/explorer/SKILL.md`. Its related notebooks/GPU recipes remain in the course repo.
- `../EECE-7373-Machine-Learning-with-Small-Data/Exercises/`: SLURM submission, interactive GPU,
  monitoring, agentic workflow, and W&B tutorials; see the course README for notebook names.
- `../README-env.md`: the separate local coursework/Jupyter environment, not FiDIP CUDA setup.

Some course examples mention files absent from this checkout (alternative Conda recipes,
`gpu_test.py`, and demo scripts). GPU recipes are generic examples, not validated FiDIP pins.
Use the actual checked-in entry points and this project's uv specification instead.

## Current readiness (2026-10-22 review)

- Setup/docs, lockfile, and environment-check scripts were published to Jack's fork and
  synchronized on Explorer at `8f4c497`. Check Git history for later documentation updates.
- Latest read-only Explorer inspection found `uv.lock` and an installation log listing installed
  dependencies; no `uv sync` process or queued jobs was observed. GPU readiness is unverified.
- SyRIP data and weights remain missing per the manifest; baseline evaluation is not started.
- Before training, verify NMS GPU-target compatibility, pretrained initialization, domain-adaptation
  settings, and full checkpoint/resume behavior. See `SKILL.md` for the specific code findings.
- Next: confirm `/projects/aclab` is the instructor-approved allocation and agree on a per-user
  path before environment changes or GPU validation. Effective access is verified, but no project
  directories were created. Obtain data/weights and reproduce the evaluation baseline afterward.

---

## Lab notebook

This is the chronological record of agent-assisted work, experiments, decisions, failures, and
changes in direction. Add entries while work happens, not retroactively at submission time.

## Logging convention

For each experiment, record:

- date, hypothesis, and one primary variable being tested
- Git commit and config path
- dataset/checkpoint inputs
- exact command and SLURM/W&B run IDs
- AP/AP50/AP75/AR and relevant training curves
- conclusion and next action

Never include credentials or `.env` values.

---

## 2026-09-21 — Project initialization

### Request

Prepare Jack's newly forked and cloned FiDIP repository for the required agentic research workflow.

### Actions

- Confirmed the clean checkout and fork remote:
  `git@github.com:jackgladowsky/Infant-Pose-Estimation.git`.
- Reviewed `README.md`, `fidip_env.yml`, the HRNet experiment config, training entry point, and
  native-extension Makefile.
- Added secret, dataset, checkpoint, log, output, W&B, and SLURM-output exclusions to `.gitignore`.
- Added a value-free `.env.example`.
- Created the required `SKILL.md`, `AGENT.md`, and `MANIFEST.md` project artifacts.

### Initial decisions

- Use the Mac checkout for code work and Explorer for Linux/CUDA compilation, evaluation, and
  training.
- Reproduce the published pretrained HRNet FiDIP result before running improvement experiments.
- Treat **93.6 AP** as the published HRNet-W48/DarkPose FiDIP reference, while retaining the full
  metric vector for comparisons.
- Keep datasets and checkpoints outside Git and record their actual locations in `MANIFEST.md`.

### Risks and open questions

- Explorer username, SSH alias, SLURM account/partition, and persistent storage paths are unknown.
- SyRIP data and pretrained model downloads require manual access and are not present.
- Local Conda is unavailable; the checked-in environment must be tested on Explorer.
- `fidip_env.yml` mixes CUDA 12.1 runtime packages with a `cuda-version=12.5` pin, so dependency
  resolution and cluster-driver compatibility must be verified before altering the environment.
- The HRNet YAML's default `TEST.MODEL_FILE` contains a Unicode multiplication sign in
  `w48_384×288.pth`, while the README uses `w48_384x288.pth`. Baseline commands should override
  this path explicitly until the discrepancy is fixed and tested.
- A project-specific SLURM script cannot be finalized until Explorer resource facts are known.

### Next action

Configure and verify passwordless Explorer access, then inventory available CUDA/modules and choose
persistent paths for data and checkpoints.

## 2026-09-21 — Explorer SSH configuration

- Jack supplied the Explorer username `gladowsky.j`.
- Added `Host explorer` to local `~/.ssh/config`, preserving existing host entries and reusing
  `~/.ssh/id_ed25519`; no key was generated or overwritten.
- Verified alias resolution to `login.explorer.northeastern.edu` with the supplied username.
- Tested `ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes explorer 'hostname; id -un'`.
- Connection reached authentication but exited 255 with
  `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)`.
- No cluster files were changed. Next: Jack should verify interactive account access and authorize
  the existing public key through the cluster-supported process, then retry non-interactive access.

### SSH retest — successful

- After Jack requested a retry, repeated the same strict-host-checking, batch-mode SSH test.
- Exit code 0; remote hostname `explorer-01`, user `gladowsky.j`.
- Non-interactive access now works. No cluster files were changed.
- Updated `SKILL.md` to reflect verified access.

## Explorer home setup

- Jack approved home storage while course project storage is inaccessible.
- Cloned Jack's fork via HTTPS to `/home/gladowsky.j/Infant-Pose-Estimation` at
  `a76b4f418107b8d3548d9779e94702e2508fbef1`; added the official `upstream` remote.
- Created empty `data/`, `models/`, `output/`, `log/`, and `slurm-logs/` directories.
- Added `slurm-logs/` to `.gitignore`; prepared the local setup files for copying to Explorer.
- Home quota remains 75 GB; filesystem-wide free capacity is not evidence of personal quota space.
- No datasets downloaded, dependencies installed, commits pushed, or SLURM jobs submitted.
- Next: validate a GPU-compatible environment and acquire baseline data/weights.

## uv environment setup — in progress

- Jack requested uv rather than Conda.
- Verified Explorer offers `cuda/12.1.1` (nvcc 12.1.105), GCC 11.5, and a `gpu` partition.
- Installed uv 0.12.17 to `~/.local/bin` without editing shell startup files.
- Added `pyproject.toml`, `.python-version`, `.venv`/build exclusions, and `ENVIRONMENT.md`.
- Started `uv sync` in the remote checkout, logging to `log/uv-sync.log`. The SSH tool timed out
  after 240 seconds, but inspection confirmed the remote sync process remained active (PID 3730847).
- Latest inspection showed managed Python 3.12.14 extraction in progress; no lockfile yet.
- Prepared `tools/check_environment.py` and `tools/check_environment.sbatch` for a bounded GPU check.
  No validation job submitted; do not call the environment ready until sync and GPU checks pass.

## 2026-10-22 — Local context and pre-Explorer readiness review

### Request and actions

- Jack requested the auto-loaded `AGENTS.md` name, a local readiness check, and an inventory of
  Explorer/course notes in this repository and the parent `eece7373` directory.
- Renamed the notebook to `AGENTS.md`, added a durable project-context header, and retained
  `AGENT.md` as a relative compatibility symlink because the course explicitly requires that name.
- Read the parent student guide and notebook-environment guide, plus the sibling course README,
  Explorer operations template, and per-GPU environment examples. Summarized the assignment
  requirements and linked local references without assuming those paths exist on Explorer.
- Updated `SKILL.md`, `ENVIRONMENT.md`, `MANIFEST.md`, and the README workflow pointer.
- Corrected stale installation-status notes using the prior read-only SSH inspection in this
  session: remote lockfile/log present, installed-package list recorded, no installer or queued jobs
  observed. Did not start installation, training, or validation; no remote files were changed.
- Flagged the legacy NMS GPU target, inactive adaptation/default pretrained settings, and incomplete
  domain-classifier resume state as pre-training checks. No model/config behavior was changed.

### Local validation

- Python AST parsing passed for all 34 Python files under `lib/` and `tools/`.
- `pyproject.toml` parsed successfully; Python 3.12 and the explicit Torch CUDA 12.1 index verified.
- `bash -n tools/check_environment.sbatch` passed.
- Notebook compatibility symlink and referenced project/course paths verified.
- Git ignore checks passed for secrets, data, model weights, outputs, virtualenv, and SLURM logs.
- `git diff --check` passed. Setup remains uncommitted; nothing was committed or pushed.

### Next handoff

Local context is ready for continuing work, not evidence of a functioning GPU environment.
Reconcile the remote-only lockfile and local changes without overwriting remote work, validate
Explorer's runtime in a bounded GPU allocation, obtain dataset/checkpoints, then reproduce baseline
metrics. Verify training initialization/adaptation and complete resumability before experiment chains.

## 2026-10-22 — Include course Explorer guide in this repository

- Jack requested an in-repository copy of the course's Explorer `SKILL.md`.
- Copied it unchanged to `docs/explorer/SKILL.md`; verified byte-for-byte equality with the source.
- Linked it from the root `SKILL.md` and project context; retained the project-specific guide.
- The copy is a generic reference template, not new cluster configuration. Related course notebooks
  are not copied, and existing SSH keys must not be overwritten by its setup examples.
- No Explorer connection, job submission, commit, or push was performed for this change.

## 2026-10-22 — Official Explorer policy review and setup guardrails

- Jack questioned installing uv in home and requested official policies, then asked to update the skill.
- Reviewed RC's general policies, local-installation, home-quota, and Conda environment documentation
  (linked in `SKILL.md`). Personal installs are supported; bulky research environments should use
  approved project storage. No uv-specific approval/policy was found.
- Distinguished the personal uv executable from Python package/environment/cache storage.
- Updated the root `SKILL.md`, `ENVIRONMENT.md`, and this context: dependency installs/builds run
  in compute allocations; confirm storage before more installs/downloads/moves. Ask instructor/RC
  for an interim location if project access is unavailable. Existing files remain untouched.
- Superseded earlier broad interim-home guidance for future work; historical notebook entries remain
  a record, not authorization for new installs. The vendored course template remains unchanged.
- Only local documentation was edited; no Explorer access, installation, migration, or job submission.

## 2026-10-22 — Authorized fork publication and Explorer inspection

- Jack authorized pushing the setup to his fork, then preparing Explorer. Installation/storage
  approval gates remain in effect; no dependency install, bulk download, or environment move started.
- Retrieved remote `uv.lock` via SCP and verified SHA-256 against the original. Local/remote
  dependency specifications have identical checksums; the original lockfile is preserved remotely.
- Git fetch confirmed the local branch and fork `origin/master` were aligned before publication.
- Read-only Explorer checks: connections intermittent; successfully reached `explorer-02`.
  Home permissions are `drwx------`; uv is user-local. `.venv/bin/python` points to the managed
  interpreter under the account's `.local/share/uv/python`. Runtime usability is still unverified.
- No queued jobs observed. Groups include `aclab`; SLURM accounts include `ostadabbas`,
  `eece5640.202630`, and `y.bart`. These are inventory facts, not approval to use a given allocation.
- Committed and pushed 14 setup/documentation/lockfile paths to Jack's fork as `8f4c497`
  (`Set up FiDIP research context and locked uv environment`). Local syntax/TOML/shell and
  whitespace checks passed; the lockfile resolves 52 package entries and matches the remote SHA-256.
- Preserved Explorer's original tracked/untracked setup in the named stash
  `FiDIP setup preserved before fork sync 8f4c497`, then pulled with `--ff-only`. Remote HEAD matched
  `8f4c497`, status was clean, the lockfile checksum was unchanged, and the notebook symlink resolved.
- Verified effective read/write/traverse access to `/projects/aclab` (also other group allocations);
  `/courses/EECE7398.202710` is inaccessible. Access alone is not course/PI approval to use storage.
- Next decision for Jack: confirm `/projects/aclab` is the appropriate allocation and agree on the
  per-user project path. No project directories created, environments installed/moved, or jobs submitted.
