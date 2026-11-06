# FiDIP Project Context and Agent Lab Notebook

## Start here

- Objective: reproduce the published HRNet-W48 FiDIP baseline on SyRIP Test100 (**93.6 AP**),
  then run controlled improvement experiments. Published scores are not reproduced results.
- Read `SKILL.md` before operating Explorer, changing the environment, or moving artifacts.
  Read `ENVIRONMENT.md` for uv setup and validation, and `MANIFEST.md` for inputs/results.
  The generic course Explorer guide is vendored at `docs/explorer/SKILL.md`; the root
  `SKILL.md` remains authoritative for this project's actual paths and safety rules.
- Run the agent on Jack's Mac/jacks-server and operate Explorer over passwordless SSH.
  **File transfers use `gladowsky.j@xfer.discovery.neu.edu`, never the `explorer` login alias**
  (datasets/models/logs and manual code/docs copies). `ssh explorer` is for commands and SLURM.
  The transfer host exposes the same personal home path; see `SKILL.md` for the official policy.
  Use compute allocations for installs/builds/evaluation/training; never run heavy work on login
  nodes. Prefer short `sbatch` jobs; interactive `srun` is optional for debugging. Request GPUs only
  when needed and chain checkpointed stages rather than defaulting to 24-hour jobs.
- Any A100 or V100 variant is approved. Current policy (2026-11-01): H100/H200 are also conditionally
  approved for training/fine-tuning only if eligible capacity is readily available without a long
  queue; otherwise use A100/V100 rather than waiting for a larger GPU. Check live availability
  before selecting a type; installed inventory is not free capacity. Always submit typed GRES and
  record the actual allocation. This does not broaden small smoke-check defaults or approve other types.
- Jack chose `/home/gladowsky.j` (75 GB limit), including the existing checkout and `.venv`.
  Do not use `/projects/aclab` or other group allocations or create/request a project allocation
  without approval. Inspect quota and existing dependencies in a read-only CPU job before installing.
  `/scratch` is for running jobs only; persist needed outputs to home before jobs end.
- Do not confuse the parent `eece7373/.venv` notebook environment with this project's separate
  Linux/CUDA environment. Do not run this project's `uv sync` on macOS.
- Only the top-level `FiDIP_Report.pdf` is eligible for report-related publication; the entire
  `report/` directory stays local. This is reproduction using the authors' provided code/model, not a new
  architecture or backbone comparison. No additional experiments or bonus claims are planned.
- Keep secrets in gitignored `.env` files; never expose their values. Keep datasets and weights
  outside Git. Confirm before canceling jobs or deleting artifacts.
- Inspect both local and remote diffs before synchronization; preserve/review `uv.lock`
  (retrieved locally from Explorer with matching checksum on 2026-10-22).
  Do not blindly copy over the Explorer checkout or commit/push without user approval.
- Jack authorized publishing the reviewed lean source release: ML code, tests/configs, required
  docs, recorded results and the top-level final PDF. This supersedes the earlier documentation-only
  restriction. Stage only named, reviewed paths; never `git add .` or `git add -A`. Do not add local
  authoring assets, data/weights, full audit logs or compiled binaries. Preserve the Explorer runtime.
- Record hypotheses, exact commands, configs, code revisions, inputs, run IDs, metrics, failures,
  and conclusions here as work happens. Update `MANIFEST.md` for produced/acquired artifacts
  and the active-runs table in `SKILL.md` for jobs.

## Course requirements and nearby references

The course requires baseline reproduction, controlled attempts to improve accuracy (including
failed attempts), and a final report covering baseline results, experiments, best configuration,
and reflection on the agentic workflow. Root deliverables are `SKILL.md`, `AGENT.md`, and
`MANIFEST.md`. `AGENTS.md` is the sole canonical context/notebook. Jack explicitly requested
removal of the former `AGENT.md` compatibility symlink on 2026-10-28. In the user-authorized
2026-11-05 final-review cleanup, an ordinary short `AGENT.md` entry-point document was added
for the course requirement; it links here and is not a restored symlink or duplicate notebook.

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

## Current readiness (2026-11-01)

- Setup/docs, lockfile, and environment-check scripts were published to Jack's fork and
  synchronized on Explorer at `8f4c497`. Check Git history for later documentation updates.
- CPU inspection job `11305792` produced PASS for the existing Python/torch/torchvision/numpy
  imports, tiny CPU matmul, and dependency consistency. Home quota reports 0.01 / 0.07 TiB
  used/limit (rounded). No reinstall needed. Revised CPU/V100 smoke checks passed (`11308338`,
  `11308377`), then full data validation passed. Original pretrained Test100 evaluation reproduced
  published rounding in `11450265`: 93.6071706887983 AP percent (published 93.6). A diagnostic
  training/resume pilot passed (`11553500`). Three prespecified tests now completed in `11553889`:
  head-only AP93.54721, stage4+head AP93.19864, original flip-off AP91.91587. Two follow-up
  diagnostics (`11556720`) also completed: BN restoration AP93.35239, fixed crop-TTA AP93.32060.
  Original93.60717 remains best. No active jobs; stop after the agreed bounded tests.
- Both SyRIP ZIPs were moved from Downloads into local gitignored `data/archives/`, extracted
  intact under `data/raw/`, and linked as `data/syrip/`. All 7,657 images decode; original Test100
  is baseline-ready (100 samples). Training is NOT yet clean: 46 dimension mismatches in the
  processed bundle and 4 byte-identical train/Test100 overlaps in both train releases. No labels
  edited. See `MANIFEST.md` / `data/validation-report.json`; full validator exits 1 for these caveats.
  The full data tree has now been copied to Explorer personal home through the required xfer
  host and CPU validation `11449533` COMPLETED `0:0`: all archive/image/annotation hashes match,
  and Test100 is ready on Explorer. Original HRNet checkpoint is now acquired and safely verified;
  baseline `11450265` COMPLETED `0:0`. Training caveats persist.
- Before training, verify NMS GPU-target compatibility, pretrained initialization, domain-adaptation
  settings, and full checkpoint/resume behavior. See `SKILL.md` for the specific code findings.
- Jack confirmed the existing personal home checkout, not lab/group project storage. Initial GPU
  validation ended TIMEOUT with a CPU-NMS dtype failure (CUDA matmul passed). Local compatibility
  fixes are synced and compiled CPU-NMS retest `11308338` passed all fixtures (script output).
  Revised typed-V100 check `11308377` COMPLETED `0:0`, all checks PASS, no longer queued.
  Data, weights, and baseline evaluation are in place. Source publication was subsequently
  authorized separately; historical entries retain the publication status at run time. User approved a
  separate 1,150-record exclusion selection for the completed three-test screen; originals unchanged.
  Next: interpret/write up the results, not launch extra training without a new request.
  SLURM dependencies sequence jobs; Bayesian optimization also requires an optimizer proposing trials.

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
- At this point storage was still awaiting confirmation. No project directories were created,
  environments installed/moved, or jobs submitted; see the later home-storage decision below.

## 2026-10-22 — Home storage and remote-agent workflow confirmed

- Jack confirmed `/home/gladowsky.j` as the chosen storage, with a 75 GB limit. No lab/group
  allocation is authorized; do not use `/projects/aclab` or create/request an allocation on our own.
- Captured Jack's notes: `/projects` is performant/persistent in general, `/scratch` is for running
  jobs only, use only necessary walltime, checkpoint and chain short jobs with dependencies.
- Agent runs on Mac/jacks-server and uses passwordless SSH to submit/monitor Explorer work.
  Routine setup can use batch jobs; interactive compute is optional for debugging.
- Bayesian optimization requires a proposal/metrics loop in addition to SLURM dependencies.
- Updated `SKILL.md`, `ENVIRONMENT.md`, `MANIFEST.md`, and this context. First proposed job is
  CPU-only quota/environment inspection with no installation. No remote work was performed in
  this documentation turn; these changes are local and not yet committed/pushed.

## 2026-10-22 — Authorized CPU quota/environment inspection

- Jack asked to run the check. Added `tools/inspect_environment.sbatch`: partition `short`,
  constraint `ib`, 1 CPU, 2 GB memory, 5-minute maximum. No installs/downloads, GPU, builds, or cleanup.
- Script SHA-256: `046a940dc860ac799139e2236ecc39640feaeb3640e0142012d2b0cc477f1be6`;
  base checkout `7bc14bc`, with the inspection script and current home-storage docs uncommitted.
- Copied only the new inspection script to the existing home checkout. Submitted using
  `cd ~/Infant-Pose-Estimation && sbatch --parsable tools/inspect_environment.sbatch` over SSH.
- SSH connections were intermittent and the submission call timed out. Before retrying submission,
  inspected `squeue`/`sacct` and confirmed exactly one job: `11305792`, PENDING (Priority).
  Do not infer failed submission from a client timeout or submit duplicates.
- Latest successful inspection reached `explorer-01` using batch SSH with GSSAPI disabled.
- Log will be `slurm-logs/inspect-11305792.out`. Quota and CPU environment results are not yet
  available at that point; this is not a GPU validation run. No installs, data downloads, or moves.

### Inspection output received

- Job ran on CPU node `d0023`. Final script output is PASS; SLURM last still reported RUNNING,
  so terminal success/exit state is not yet confirmed. Do not cancel without Jack's confirmation.
- `check-quota "$HOME"` reported rounded 0.01 TiB used, 0.07 TiB soft/hard limit, and 33,220 inodes
  used (1,000,000 soft / 1,500,000 hard). The tool labels the backend path `/discovery/home`.
- uv 0.12.17; cache `/home/gladowsky.j/.cache/uv`; Python 3.12.14 resolved under
  `/home/gladowsky.j/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12`.
- PyTorch 2.3.1+cu121, torchvision 0.18.1+cu121, and NumPy 1.26.4 imports passed; tiny CPU
  matrix multiplication passed. All 51 installed packages passed `uv pip check`.
- No sync/reinstallation needed. GPU/CUDA execution and native extensions remain untested.

## 2026-10-22 — Authorized GPU/native-extension validation

- Jack authorized the GPU test. Confirmed CPU inspection accounting COMPLETED with exit `0:0`
  and elapsed 4m44s; controller still reported COMPLETING (cleanup), not an active computation.
- Remote tracked checkout was clean; only the earlier inspection script was untracked. GPU-check
  scripts matched local/published checksums. No uv reinstall, data download, or model training.
- Submitted `sbatch --parsable --dependency=afterok:11305792 tools/check_environment.sbatch`
  from `/home/gladowsky.j/Infant-Pose-Estimation`; GPU job ID `11305949`, partition `gpu`,
  default account `ostadabbas`, QOS `normal`, 1 GPU, 2 CPUs, 8 GB, 10-minute limit.
- CPU controller cleanup held the dependency unfulfilled despite successful output/accounting.
  These tests have no file dependency; removed it with `scontrol update JobId=11305949 Dependency=`.
  Neither job was canceled; resource requests and time limit were unchanged.
- GPU job last observed PENDING, dependency null. Output: `slurm-logs/env-11305949.out`.
  Do not claim GPU readiness until compilation, entry-point imports, and CUDA/NMS checks pass.
- Check script SHA-256 values: Python `2378de4593b676533abdd387d59a6529d76c84ffff4966d9d27bc4533f035bd9`;
  batch `b4f8ed5c2befe153063679f38c255bdea61b32d4bcc3a76251469bf6560dba73`.
- Native builds may modify tracked generated files; preserve logs and inspect remote diffs before
  syncing changes. Do not publish generated binaries. Current documentation edits remain local.

### GPU resource-selection correction

- Job ran on `d4055`; a read-only diagnostic step within the existing allocation identified
  NVIDIA H200. Generic `--gres=gpu:1` allowed this assignment; it was not justified for a smoke test.
- CPU/GPU NMS extensions compiled with CUDA 12.1.1. No entry-point/CUDA/NMS runtime PASS yet.
  Python process was observed in `Dl`, `folio_wait_bit` (file/page I/O wait), not a demonstrated
  CUDA failure. No automatic reinstall or extra GPU job was launched.
- Jack explicitly requires economical GPUs (V100 or A100), not H200. Changed the local future
  check-script default to `--gres=gpu:v100-sxm2:1` and added a no-unconstrained-GPU guardrail.
  This local edit does not alter the already-submitted script/job and has not been deployed/pushed.
- Last accounting state was RUNNING at 9m45s with the original 10-minute cap. No job canceled
  or resource/time limit increased; cancellation still requires Jack's explicit confirmation.

### Final outcome verified

- On Jack's follow-up, accounting reported job `11305949` TIMEOUT, elapsed 10m30s; it is no
  longer in `squeue`. Accounting displayed `0:0`, but TIMEOUT and the Python traceback mean
  this is not a successful test.
- Both native extensions built. Training/evaluation `--help` completed, confirming entry-point
  imports. Python reported H200, PyTorch 2.3.1+cu121, CUDA runtime 12.1; the CUDA matmul assertion
  passed because execution proceeded to the CPU-NMS assertion afterward.
- CPU NMS failed at `cpu_nms.pyx:28`: `ValueError: Buffer dtype mismatch, expected 'int_t' but
  got 'int'`. GPU-NMS execution was never reached; do not mark the full GPU environment ready.
- Source inspection found `np.int_t` order declared but `.astype('i')` supplies 32-bit integers;
  the later `np.int` alias also needs modernization for NumPy 1.26. No fix/retest launched in this
  status-check turn. Next proposed work is a minimal explicit-integer-dtype fix and V100-only retest.
- Log retained at `slurm-logs/env-11305949.out`. Documentation updated locally; no commit/push.

## 2026-10-22 — Smoke-test review, compatibility fixes, and timing instrumentation

- Jack requested a run-through, stage timings, and a GPU inventory/suitability check. Read NMS
  wrappers/kernel/build setup, the smoke scripts, SyRIP loader, and model/dataset import paths.
- Read-only Explorer query confirmed public `gpu` GRES types: V100 PCIe, V100 SXM2, A100, T4,
  H200. Snapshot had mixed/allocated/draining/completing nodes; no fully idle node. Installed
  inventory is not free GPU capacity. V100 is adequate for this test; training fit needs a pilot.
- Latest accounting for `11305949`: TIMEOUT, exit `1:0` (supersedes earlier displayed `0:0`).
  Only generated NMS files and the inspection script differed on Explorer; no source fixes deployed.
- Local CPU-NMS fix: explicit matching `np.intp_t`/`np.intp` buffers; removed deprecated scalar
  threshold aliases in both wrappers and added a GPU empty-input guard. No NMS algorithm change.
- Local native build now defaults to V100 `sm_70` plus PTX, configurable via `NMS_CUDA_ARCH`
  (80 for an explicitly chosen A100). No Torch/GPU import needed to compile on a CPU node.
- Split workflow: `tools/build_environment.sbatch` (CPU, 5m) compiles and tests CPU NMS; dependent
  `tools/check_environment.sbatch` (explicit V100, 5m) tests CUDA/NMS and project/CLI imports.
  Both request 2 CPUs and 4 GB host RAM; no package installation or data downloads.
- Added common timestamped stage/total timings, source hashes, exit-code reporting, per-stage
  timeout limits, unbuffered Python check/case timings, and periodic stack dumps during stalls.
  Independent checks continue after an exception and the process exits nonzero if any fail.
- NMS cases cover empty/single/duplicate/disjoint/mixed boxes and 129 boxes crossing 64-box
  CUDA blocks. Added local harness regressions, including refusing H200 before tensor allocation.
- Audit follow-ups: five `np.float` aliases remain in the separate COCO loader, not the active
  SyRIP loader; training initialization/adaptation and complete resume state still need review.
- Local validation: 10 harness tests ran, 9 passed and 1 skipped because GNU `timeout` is absent
  on the Mac. Bash syntax passed for all four helper/job scripts; Python AST parsing passed for
  35 source/test files; `git diff --check` passed. These are harness/static checks, not compiled
  Cython/CUDA verification. An initially incomplete mock fixture was corrected before the final run.
- Changes are local, uncommitted/unpushed, not deployed or compiled on Explorer. No new jobs,
  installs, downloads, storage moves, or generated binaries were created on the cluster in this turn.

## 2026-10-22 — Flexible A100/V100 preference and revised test authorization

- Jack asked to retain the GPU ranking and clarified that any A100 or V100 variant is acceptable;
  H200 is unnecessary. Updated current context and `SKILL.md` accordingly. Typed GRES requests
  are still required to prevent accidental H200 allocation, but no per-variant permission is needed.
- Jack asked to test the revised scripts. Plan: preserve Explorer's generated-file changes, copy
  only reviewed source/scripts/docs after checking they have no conflicting remote source edits,
  then run the 5-minute CPU build/regressions. Submit the 5-minute typed GPU check only on success.
- No package installation, training, data download, storage migration, commit, or push authorized
  by this test step. Source/test hashes and new run IDs/results will be recorded below.
- Verified Explorer had no conflicting edits to target source/docs; existing generated native-file
  changes were left in place. Synced 12 explicit source/script/test/doc paths and verified every
  SHA-256 against the local copy. No package/environment change or Git push was performed.
- Local harness tests still report 9 passed / 1 skipped on Mac. Added harness execution to the
  CPU build job so GNU-timeout behavior can be tested on Linux before compiling native code.
- Initial combined verify/submit SSH call timed out after hash verification. Checked `squeue`,
  `sacct`, and build logs: no job existed, then submitted the CPU script alone successfully.
- CPU build job `11308157`, `short`, 2 CPUs, 4 GB host RAM, 5-minute cap; last observed PENDING
  (Priority) initially. Log: `slurm-logs/build-11308157.out`. No GPU job submitted yet.
- Timed result: job `11308157` ran on `d0024`; bootstrap/compiler/source-hash stages passed,
  all 10 Linux harness tests passed (including GNU-timeout behavior), then native compilation
  failed in 3s opening `/home/gladowsky.j/Infant-Pose-Estimation/lib/nms/cpu_nms.c` for Cython output.
  The process logged `JOB_END rc=1 elapsed_s=5`; accounting still showed RUNNING during cleanup.
  This is a generated-file path failure, not a demonstrated error in the new NMS integer types.
- Read-only checks on the login node found an existing regular generated C file and normal NMS
  directory permissions; no symlink. Exact filesystem cause is not proven. Confirmed installed
  Cython 3.0.10 supports `--cython-c-in-temp` using its own build_ext source; no package install.
- Minimal build-path adjustment: generate C/C++ beneath the ignored build directory, and add
  the NMS source directory to the GPU extension's header search path. Regression asserts the
  temporary-C flag. Local tests still 9 passed / 1 skipped on Mac.
- Safely synced only the three changed build/setup/test files after checking their previous hashes.
  CPU retest `11308338` submitted via `sbatch --parsable --job-name=fidip-build-temp
  tools/build_environment.sbatch`, still 2 CPUs / 4 GB / 5m, no GPU. New script/setup/test SHA-256:
  `3ed295441b8653d6d6e11df84ea11af4ab052d3a9bc37870a64b911fd71eb3cc`,
  `8cb5c9b9f1271eb180fc8774424bd1aacc198b7984c8ceef2770ff7281ddc704`,
  `c78757b4ea7632cf9250ae0880b9e96cd8a5f1cc080efcc0f13db04ff20a12b9`.
- CPU retest ran on `d0014`: all 10 harness tests passed, both native modules compiled (`sm_70`
  plus PTX), and all 6 CPU-NMS fixtures passed. Timings: harness 2s, native build 12s, CPU checks
  <1s, script total 15s. `JOB_END rc=0` observed; terminal accounting state still RUNNING afterward.
- After positive CPU results, submitted the GPU script with explicit `--gres=gpu:v100-sxm2:1`
  and `--dependency=afterok:11308338`. First call timed out; checked queue/accounting (no child)
  before retrying. The retry stores its returned ID in
  `slurm-logs/gpu-after-11308338.submit` to survive an SSH output loss and avoid blind resubmission.
- GPU job ID `11308377`, 2 CPUs, 4 GB, 1 V100-SXM2, 5-minute cap, default account `ostadabbas`.
  Last controller state PENDING, dependency `afterok:11308338` unfulfilled while parent finalizes;
  typed request verified via `TresPerNode=gres/gpu:v100-sxm2:1`. No GPU allocated/runtime result yet.
- Logs: `slurm-logs/build-11308338.out`, `slurm-logs/env-11308377.out`. Source remains uncommitted
  over `7bc14bc`; job hash logs identify exact inputs. No installs, training, data downloads, or push.
- Final CPU accounting now confirms COMPLETED `0:0`, allocation elapsed 4m56s; controller CG
  cleanup still held the GPU dependency at the last check. Keep allocation elapsed separate from
  the measured 15s script runtime; infrastructure/startup/cleanup overhead is not measured by the
  script's stage timer. At that point GPU job `11308377` remained queued; see final result below.

## 2026-10-22 — Queue monitoring and successful V100 validation

- Jack requested queue monitoring and completion notification. Polled `squeue`/`sacct` over SSH
  approximately every 30–40s, without allocating compute or submitting more jobs. Dependency
  cleared, job waited on priority, then ran on `d1007`.
- Verified terminal accounting for `11308377`: COMPLETED, exit `0:0`, allocation elapsed 2m58s;
  `squeue` is empty for this job. Final log contains PASS and `JOB_END rc=0`, script elapsed 26s.
- Actual GPU: Tesla V100-SXM2-32GB, NVIDIA driver 545.23.08, capability 7.0, Torch-reported memory
  31.7 GiB. Runtime: Python 3.12.14, Torch 2.3.1+cu121, torchvision 0.18.1+cu121, CUDA 12.1.
- All checks passed: CUDA availability/matmul, CPU NMS (6 cases), GPU NMS (6 cases), torchvision,
  dataset, models, TensorBoard imports, and training/evaluation CLI help. No training or data access.
- Timings: combined GPU/project checks 17s, training CLI 5s, evaluation CLI 4s. Within Python:
  CUDA/import+matmul 8.402s, CPU NMS 0.042s, GPU NMS 0.008s, torchvision import 5.644s,
  dataset import 0.500s, models import 0.028s, TensorBoard import 0.890s. These are smoke timings.
- Updated current readiness, operations, environment docs, and manifest locally. No new installs,
  downloads, storage moves, commits/pushes, or jobs. Next is SyRIP/checkpoint acquisition and
  baseline evaluation; full training/model metrics are not implied by a successful smoke test.

## 2026-10-22 — Dataset availability verified online

- Jack asked whether dataset acquisition needs his intervention. Fetched official SyRIP listings
  and README; checked ZIP metadata via bounded HTTP ranges, without full archives or image payloads.
- Complete central-directory inspection found no encrypted files in original `SyRIP.zip`
  (327,912,103 bytes) or `syrip_for_train.zip` (637,172,858 bytes). Training ZIP includes the
  pre-infant, infant train, and validation annotation filenames expected by this repository.
- Original ZIP includes separate Test100/Test500 annotation trees; select Test100 explicitly
  for comparison with published 93.6 AP. Inspect actual counts/schema after download; do not
  assume the flat training-archive validation file uses the desired split.
- Expanded SyRIP ZIP metadata sample had 500/500 file entries encrypted; would need an authorized
  password if choosing that variant. It is not needed to start the original baseline workflow.
- Updated current access facts in `SKILL.md` and `MANIFEST.md`. No full dataset download, Explorer
  job, package install, commit, or push performed during this availability check.

## 2026-10-28 — Local dataset move, extraction, and full asset validation

- Jack confirmed both downloads and authorized moving them into the repo, extracting, and checking
  readiness. Moved `~/Downloads/SyRIP.zip` and `syrip_for_train.zip` to gitignored `data/archives/`.
  Original download paths no longer exist; archives are retained intact in the repository data area.
- Inspected ZIP paths/encryption/case-folded collisions/symlinks before extraction. Separate raw
  roots preserve case-distinct `SyRIP` and `syrip` trees safely on macOS without collisions/overwrites.
  CRC checks passed during extraction, and archive SHA-256 values are recorded in `MANIFEST.md`.
- Built `data/syrip` using relative symlinks: FiDIP train/pre-infant inputs, original Test100 labels
  and original validation images. Kept both raw archives/extractions, all train variants, and
  Test500 untouched. Transfers must retain targets as well as symlinks.
- Added ignored `data/validate_datasets.py` and reports. Fully decoded/hashed all 7,657 JPEG files,
  checked every source annotation plus active paths/IDs/categories/17-joint shape/finite values,
  dimensions, loader filtering, Test100 consistency, and exact-byte train/validation overlaps.
- Baseline Test100 passes: 100 images/annotations, no missing or dimension-mismatched files,
  valid keypoint order/schema. Training ZIP's Test100 IDs/core labels match the original, but its
  `test11.jpg` is differently sized; working links use the consistent original validation images.
- Pre-infant: 4,057 records/files, 153 invalid boxes filtered by the existing loader, leaving
  3,904 usable (1,876 real / 2,028 synthetic). This is expected loader filtering, not corruption.
- Training: 1,200 referenced samples (200 real / 1,000 synthetic) among 1,500 files. Found 46
  JPEG/annotation dimension mismatches in processed training inputs. Original pairing has none,
  but synthetic keypoint/bbox annotations differ between releases; do not silently swap protocols.
- Found four byte-identical train/Test100 pairs in BOTH original and processed training sets:
  train00272/test46, train00073/test20, train00315/test42, train00340/test50. No exact-byte
  pre-infant/Test100 overlap. Near-duplicate similarity was not tested.
- Preserved labels and all warning details (num_keypoints metadata, out-of-frame pre-infant
  coordinates). Did not remove duplicates or automatically rescale/repair annotations.
- `data/validation-report.json`: baseline_test100_ready=true, training_ready_without_review=false;
  validator intentionally exits 1 to flag unresolved training quality issues. Local footprint
  about 1.81 GiB. Added `data/README.md` and updated manifest/operations/current readiness.
- No existing model/config/source changes, Explorer transfer/jobs, downloads, installs, commit,
  or push performed. Existing unrelated working-tree changes remain intact. Pretrained weights
  are still needed for baseline evaluation; training needs an explicit repair/leakage decision.

## 2026-10-28 — Explorer transfer route correction and persistent guardrail

- Jack authorized preparing data on Explorer. Read-only inspection confirmed existing remote
  source/generated changes, no queued jobs, and an empty `data/` target; did not reset or pull Git.
- Fresh CPU quota job `11448779` COMPLETED `0:0` in 1s on `d0014`: 0.01 TiB used / 0.07 TiB
  soft/hard limit (rounded), 35,720 inodes used. Enough room for the approximately 1.81 GiB tree.
- Copied immutable local image/report snapshots (`*.local.json`) into ignored data and prepared
  CPU-only transfer-verification helpers. No package installation, GPU request, or training.
- Initial rsync attempts incorrectly used the Explorer login endpoint and returned EOF, leaving
  partial receiver files. Then fetched the official RC Transfer Data guide, which explicitly
  requires `xfer.discovery.neu.edu` for local↔cluster file copies. Stopped login-node transfers.
- Verified passwordless transfer-host access as `gladowsky.j` on `xfer-00` with the existing
  ed25519 key and StrictHostKeyChecking=yes. Same personal-home destination is visible there.
  No SSH configuration/key changes or new project allocation were made.
- Corrected rsync to `gladowsky.j@xfer.discovery.neu.edu:/home/gladowsky.j/Infant-Pose-Estimation/data/`
  with `-rltp --partial --timeout=120 --bwlimit=20000 --stats`, no --delete. Final copy exited 0
  in 273s; source tree 1,948,049,676 bytes / 7,705 entries, 924,613,109 remaining bytes transferred.
  Relative symlinks and original/raw inputs retained. CPU cryptographic/image verification is
  still pending; successful rsync alone is not the final readiness check.
- Receiver log: `log/data-xfer-rsync-20261028.log`. Earlier partial transfer file
  `data/archives/.syrip_for_train.zip.RdQRPX` (about 403 MiB) may remain; verify final archive
  checksums before cleaning up this known transient, and do not delete unrelated artifacts.
- Jack explicitly requested a durable note to avoid repeating the route mistake. Added prominent
  xfer-vs-login rules and official-source examples to project SKILL/context and global memory.
  This supersedes earlier file-copy assumptions; all future manual code/docs copies also use xfer.
- At this stage no remote dataset validation job had been submitted. See the subsequent
  successful validation entry; weights, model evaluation, training repairs, and commits/pushes
  remain separate tasks.

## 2026-10-28 — Explorer dataset validation completed

- Jack requested validation. Rechecked all six local/remote script and input-audit SHA-256 values
  and inspected working relative links; confirmed no existing data-validation job before submitting.
- Submitted `tools/validate_dataset.sbatch` as `11449533`, via the Explorer command endpoint,
  persisting the ID in `slurm-logs/data-validation-20261028.submit`. Resource request: `short`,
  `ib`, 4 CPUs, 4 GB RAM, 5-minute cap, no GPU. No installs, training, or dataset modifications.
- Monitored the existing job rather than resubmitting. Verified COMPLETED, exit `0:0`, out of
  `squeue`; host `d0005`, allocation elapsed 31s, script elapsed 28s, checker elapsed 27.663s.
- Both ZIP hashes/sizes match the Mac archives (hash phases 1.161s and 2.288s). All 7,657 images
  fully decode and match source SHA-256/dimensions. All raw/working annotation hashes match.
- Original Test100 is ready on Explorer: 100 images/annotations, no missing or dimension-mismatched
  references. Pre-infant has 4,057 records / 3,904 loader-usable samples; infant training has 1,200.
- Known issues are unchanged: 46 processed-training dimension mismatches, 4 train/Test100
  byte-identical overlaps, plus documented metadata warnings. Raw auditor exit 1 is intentional;
  the transfer wrapper exits 0 only for an identical copy and passing Test100. Training remains
  not approved/clean. No labels or splits were edited to force a pass.
- Remote summary: `data/explorer-transfer-validation.json`; log: `slurm-logs/data-11449533.out`.
  Retrieved both through strict-key/passwordless xfer SSH into local
  `data/explorer-validation/2026-10-28/`, not through the login node.
- Backed up and inspected remote context before synchronizing current operations/results;
  remote-only lines were stale versions of known setup notes, not new user/source changes.
  Backups live in the ignored local evidence folder. Existing native/source changes are preserved.
- At this point the original FiDIP checkpoint was still missing. No GPU/model evaluation, data
  cleanup, commit, or push occurred in that validation turn; see the later model/baseline entry.

## 2026-10-28 — Original HRNet acquired; published Test100 baseline reproduced

- Jack supplied `~/Downloads/hrnet_fidip.pth` after the official original FiDIP model folder was
  linked. Moved it unchanged into gitignored `models/`; file is 255,354,649 bytes, SHA-256
  `bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07`.
- Local ZIP CRC and `torch.load(weights_only=True, map_location='cpu')` passed. Raw OrderedDict
  has 1,754 tensor entries; all HRNet keys and shapes match. No author-published digest was
  available; source provenance is the user's original-folder download, not independently proven.
- Safely synced only reviewed evaluation script changes/new batch files and checkpoint/metadata
  through xfer. Checked remote preimage before modifying the entry point; unrelated source/native
  changes were preserved. No Git reset/pull, environment install, training, or data repairs.
- Replaced evaluation's unsafe/lenient load with tensor-only CPU loading and `strict=True` so
  missing/extra weights cannot silently produce a misleading score. For this exact matching
  checkpoint, model/dataset inference math remains unchanged.
- CPU admission `11450222` COMPLETED `0:0`, `d1027`, 13s allocation / 12s script; finite tensors,
  strict load, SHA-256, and CPU forward passed. Heatmaps `[1,17,96,72]`, features `[1,48,96,72]`.
  CPU report: `log/checkpoint-11450222.json`, copied locally under `models/validation/` via xfer.
- Submitted `tools/evaluate_baseline.sbatch` as `11450265`, afterok `11450222`, exporting
  `FIDIP_CHECKPOINT_JOB=11450222`. Typed V100-SXM2 request, 4 CPUs, 8 GB host RAM, 5-minute cap.
- GPU run COMPLETED `0:0`, no longer queued, on `d1020` with V100-SXM2-32GB / driver 545.23.08.
  Allocation 71s; script 64s; baseline evaluation stage 60s; full-precision COCO rescore 1s.
- Original Test100, GT boxes, flip and postprocess enabled, batch 16, 384x288 input, official YAML
  unchanged. Only debug rendering (disabled), log frequency, and per-job output/log paths changed.
  Config SHA-256 `cd1151abad97aac80400252c89e000c3ea7fca87975f87b06546e66cc1191157`;
  annotation SHA-256 `8b299a879a95913483498e3f2115a826ac23ebc5262c3efb8d65f63718e3194c`.
- Full metrics (fractions): AP 0.936071706887983; AP50/AP75 0.9852475247524751; AR 0.946;
  AR50/AR75 0.99. AP is **93.6071706887983 percent**, matching published **93.6** after rounding,
  not an improvement over the reference. Medium-area stats -1 mean no applicable medium examples.
- Metrics: `output/baseline-test100-11450265/metrics.json`; nested results JSON and upstream
  feature.npy/log retained on Explorer. Local metrics/prediction copies and CPU/GPU logs retrieved
  via xfer under `output/baseline-test100-11450265/`, `models/validation/`, and
  `log/baseline-evidence-11450265/`. Source/script hashes are preserved in those logs and manifest.
- No training started. The 46 processed-training size mismatches and four original train/Test100
  overlaps remain unresolved and must not be silently removed or waved away by reproduction.
- Updated notebook/manifest/operations/environment records and model/data metadata. Current source
  is still uncommitted reviewed changes over `7bc14bc`; no commit or push was performed.

## 2026-10-28 — Documentation-only publication and notebook alias removal

- Jack authorized publishing `AGENTS.md`, `MANIFEST.md`, and `ENVIRONMENT.md`, but explicitly
  excluded compatibility fixes and runtime/tooling changes. These remain in the working checkouts;
  the published docs do not claim a fresh clone reproduces the validated runtime unchanged.
- Removed only the tracked `AGENT.md` symlink at Jack's request; preserved the actual `AGENTS.md`
  notebook and all historical entries. Updated current filename references and the README note.
  The singular course filename can be reconsidered for final submission, not silently restored.
- Commit scope: the three documentation files, the small README reference/publication notice,
  and deletion of `AGENT.md`. No `SKILL.md`, source fixes, tests, scripts, generated binaries,
  datasets, checkpoints, logs, or output payloads are included.
- Publication does not authorize a remote checkout reset/pull, alias deletion on Explorer, or new
  jobs. Preserve the working Explorer runtime and its uncommitted changes during later syncs.

## Fine-tuning screen proposal and conditional Hopper permission (current as of 2026-11-01)

- Discussed five independent short continuations from the verified FiDIP pose checkpoint: natural-
  mixture low-LR control, 1:1 real/synthetic sampling, final-stage/head-only updates (earlier layers
  and their BatchNorm stats frozen), mild photometric augmentation, and mixed-to-real curriculum.
  These are proposals, not launched runs. Training-data review and initialization/resume validation
  remain prerequisites. Keep the Test100 protocol fixed and establish development validation.
- Planning budget is roughly 500–1,000 optimizer steps per run with equal-step comparisons;
  about 1–3 GPU-hours for the first five-run screen is unmeasured, not a throughput guarantee.
  First measure step time/VRAM in a short pilot; preparation, queueing, and repeat seeds are extra.
- Jack explicitly permits H100/H200 for training/fine-tuning if available without a long wait.
  Prefer readily available eligible capacity; otherwise fall back to A100/V100. No live Hopper
  availability/compatibility has been established. Historical H200 smoke-test restrictions remain
  the record of earlier decisions, not a prohibition on this newly approved training use.
- Updated local current GPU policy in AGENTS/SKILL/environment notes only. No commit/push,
  remote synchronization, job submission, cancellation, or training was performed for that policy update.

## 2026-11-01 — Training audit and diagnostic pilot

- Jack authorized proceeding with a short pilot and conditionally permits H100/H200 when available
  without a long wait. Runtime changes stay unpublished; no new Git commit/push is authorized.
- Read the actual training path before experiments. Correction to the earlier sampling proposal:
  `BalancedBatchSampler` already alternates real/synthetic 1:1; the raw 200/1,000 record ratio is
  not the author's effective batch ratio. Do not claim changing to 1:1 as a new improvement.
  The author script also freezes the first 109 parameter tensors by index but not their BN stats;
  restart omits domain optimizer/scheduler and RNG/sampler state. Lambda zero still runs a random
  domain classifier, so the diagnostic explicitly uses supervised pose loss only.
- Added unpublished `tools/audit_finetune_readiness.py`, `tools/audit_finetune.sbatch`,
  `tools/pilot_finetune.py`, `tools/pilot_finetune.sbatch`, and `tests/test_finetune_pilot.py`.
  Three CPU unit tests cover RNG restoration, exact model/Adam continuation, protocol mismatch,
  checkpoint overwrite refusal, and state fingerprints. No new dependencies installed.
- CPU audit `11553238` COMPLETED `0:0` on `d1027`: script 25s, allocation accounting 92s.
  Rehashed model/train/Test100 annotations and image payloads against the preserved Mac manifest.
  Unchanged findings: 46 synthetic dimension mismatches and 4 real train/Test100 byte overlaps.
  Candidate unflagged population: 1,150 (196 real, 954 synthetic), not yet an approved experiment split.
- Admission selects only 16 real + 16 synthetic verified, non-overlapping examples for a diagnostic
  smoke test. No raw/working annotations, labels, images, or splits were edited/deleted. The pilot
  is not a five-experiment result and does not score Test100. Explicit approval is still needed
  before excluding the flagged 50 samples in scientific comparison runs and fixing validation policy.
- CPU report `output/finetune-readiness-11553238.json`; stdout `slurm-logs/train-audit-11553238.out`.
  Retrieved both via xfer to `log/finetune-evidence-2026-11-01/`.
- Live H200 inventory showed spare GPUs but scheduling forecasts were long; no H200 job submitted.
  Selected V100 on public `gpu-short` (2h partition maximum), explicit `gpu:v100-sxm2:1`, 4 CPUs,
  8 GB host RAM override, 5m job limit, and deadline now+7m (latest start within about 2m).
  Submitted diagnostic `11553500`; started within ~34s on `d1019`, V100-SXM2-32GB.
- Diagnostic plan: 50 FP32 optimizer updates, batch 8, Adam LR 1e-5, deterministic algorithms,
  TF32 off, existing geometry/normalization/loss, workers=0 for reproducible augmentation RNG.
  Save pose+Adam+RNG+global step/protocol, compare 3 uninterrupted steps against 3 restored steps.
  Original checkpoint is hashed before/after and never overwritten. New pilot checkpoint stays home.
- Pilot `11553500` verified COMPLETED `0:0`, out of queue: allocation accounting 135s, script
  56s, Python body 44.277s. All 50 training updates had finite loss/gradients; head weights changed.
  Both 3-step branches matched exactly across pose/Adam/Python+NumPy+Torch+CUDA RNG state.
  Source checkpoint hash unchanged. This validates the new diagnostic path, not the author's
  incomplete adversarial-resume implementation and not any AP improvement.
- Median steady step 0.3895523s at batch 8; compute-only extrapolation 500 steps 3.246m,
  1,000 steps 6.493m. Peak allocated GPU 4.299 GiB, reserved 4.990 GiB; batch MaxRSS 1,492,904 KiB
  (~1.424 GiB). Includes exhaustive gradient checks; tiny-subset timings do not establish full-data
  I/O/validation/queueing time. V100 is adequate; no reason to wait for Hopper for this screen.
- New `output/finetune-pilot-11553500/resume.pth` is 764,835,130 bytes, diagnostic pose+Adam+RNG
  state at step 50, NOT a raw pose-only/best-model file for the evaluation CLI. It remains on Explorer.
  `protocol.json` and `pilot-report.json` are beside it; stdout `slurm-logs/train-pilot-11553500.out`.
  Retrieved reports/protocol via xfer to local `output/finetune-pilot-11553500/`; stdout/accounting
  to `log/finetune-evidence-2026-11-01/`. Both audit/pilot jobs completed, no active jobs remain.
- Proposed next decision, not yet applied: separate derived training profile excluding the 46
  inconsistent examples and 4 exact Test100 overlaps, retaining 1,150 records (196 real/954 synthetic),
  originals unchanged. Need explicit approval before these protocol-changing exclusions.
- Revised five-run proposal: balanced 1:1 full-network control; 2:1 real-heavy; final-stage/head
  only with frozen early BN stats; mild photometric augmentation; mixed-to-real curriculum.
  A 600-step/batch-8 screen would allow the curriculum's first 400 steps balanced, last 200 real-only,
  matching the total 3,200 real / 1,600 synthetic exposures of the fixed 2:1 trial. All runs start
  independently from the original weights. No Test100 early stopping; choose/approve a development
  and final-evaluation protocol explicitly before comparison runs. Prior exposure in released
  weights remains a limitation even after exclusion from new fine-tuning.
- Runtime source hashes are in job logs. No source/doc push, remote Git reset/pull, job cancellation,
  full-data comparison training, label repair, source data removal, or Test100 model selection.

## 2026-11-01 — Approved three-test screen (supersedes five-run proposal)

- Jack cannot access expanded SyRIP and prefers minimal data-curation effort. After the automatic
  exclusion list was explained, he approved the three proposed tests and then asked to continue.
  This authorizes a separate 1,150-record selection manifest excluding 46 inconsistent-dimension
  examples and 4 exact Test100 overlaps. Source releases, images, labels and Test100 remain unchanged.
- Prespecified tests: (1) final prediction head only, (2) stage4 + final head, each independently
  initialized from original FiDIP weights, 600 optimizer updates, batch8, Adam LR1e-5, seed7373,
  exactly balanced real/synthetic sampling with replacement. Same step-indexed image batches and
  geometric augmentation seeds; frozen modules/BatchNorm stats remain in eval mode and are hashed
  before/after. No new domain classifier/adversarial objective; no Test100 early stopping.
- (3) Disable flip testing on the ORIGINAL checkpoint, not a tuned checkpoint, to isolate the
  inference tradeoff. Recheck original flip-on AP first; all evaluations use existing repo dataset,
  decoder, flip-back, OKS rescoring/NMS and COCO metric code. Paired warmed GPU-forward timings
  explicitly exclude preprocessing, transfers, flip-back and decoding; do not call this app speedup.
- Added unpublished `tools/prepare_small_screen.py`, `tools/prepare_small_screen.sbatch`,
  `tools/run_small_screen.py`, `tools/run_small_screen.sbatch`, `tests/test_small_screen.py`.
  Local four screen tests and three prior resume tests passed, plus Python/Bash syntax and diff checks.
  Runtime files transferred via xfer only, after asserting destination files did not already exist.
- CPU prep `11553804` submitted once; all seven Linux tests, fresh payload/hash checks and approved
  selection serialization passed. Script JOB_END rc0 after 18s; allocation subsequently reported
  TIMEOUT with exit0:0 after 5m08s. Preserve this discrepancy; do not call the allocation COMPLETED.
  Slurm remained RUNNING long after the successful script ended; cause not established. No duplicate
  prep or job cancellation. Logs `slurm-logs/screen-prep-11553804.out`, approved manifest
  `output/small-screen-profile-11553804.json`; xfer copies in `log/small-screen-evidence-2026-11-01/`.
- After directly verifying successful CPU JOB_END and complete approved manifest, submitted GPU
  `11553889` without a Slurm dependency on the anomalous CPU allocation. GPU independently verifies
  all pinned inputs. One bounded 15-minute allocation, gpu-short, typed V100-SXM2, 4CPU/8G, deadline
  now+17m; periodic 300/600-step checkpoints and immediate per-test results. Started on d1022.
  No Hopper job submitted. GPU run now verified COMPLETED `0:0`, out of queue, allocation 6m45s,
  script 6m18s, experiment stage 6m16s. Actual V100-SXM2-32GB on d1022; batch MaxRSS 3,476,404 KiB.
  Original flip-on rechecked exactly 93.6071706887983 AP. All three tests completed successfully.
- Head-only: AP 93.5472078062278, delta -0.05996288257050253 AP points, AR94.5, 833 trainable
  parameters. Stage4+head: AP93.198637527681, delta -0.40853316111730464, AR94.3, 45,894,881
  trainable parameters. Both completed 600 updates; frozen parameters/BN buffers hash-identical.
  Training-step sums 126.514s / 166.470s; median steps 0.20023s / 0.27117s, respectively.
- Original weights flip-off: AP91.9158670222938, delta -1.6913036665045098 AP points, AR93.4.
  Nine paired warmed forward repeats at batch16: flip on 13.00666 ms/image, off 6.50483 ms/image,
  speedup1.99954x. This is CNN-forward ONLY, excluding I/O/preprocess/transfers/flip-back/decoding;
  do not claim 2x application speedup. Single-pass wall times 5.779s/3.516s are cache/order affected.
- Neither fine-tune improved this one-seed screen; head-only is numerically close and no statistical
  significance was established. Keep original weights as best tested checkpoint. No further jobs
  are active or implied. These exploratory comparisons do not establish independent generalization
  or show that every fine-tuning strategy fails; initial checkpoint prior exposure remains a limit.
- Outputs `output/small-screen-11553889/{results.json,protocol.json,summary.md}`; per-case predictions
  and head/stage4 final pose checkpoints plus 300/600-step resume states retained on Explorer.
  Head final SHA256 `9a8f03f65328baceb05ce36a111d9d7b71337d182e4a59dbfa241ee4e5a667ff`;
  stage4 final `b146f8c25d5c45f882603f858d1c532d054e1caedc687fedb41e154fd3be3f6e`.
  Profile SHA256 `42c193dc87f62eb0ba1aba60544bd95e4a4575d651c80c0320a05fde9cf6edc7`.
  Runtime runner SHA256 `3c44a515adbef57b2f913ab235ad27096c430eded53a705f7fa6227b92f0e722`.
- Stdout/accounting/prep/profile logs copied via xfer to `log/small-screen-evidence-2026-11-01/`;
  JSON results/protocols/predictions copied via xfer under the same local output run folder.
  Large weights/resume states remain Explorer home only. No new source/doc push; Git HEAD remains
  documentation commit `bc66664`. Source data and original checkpoint unchanged.
- Single-seed fixed-budget comparisons are exploratory. Excluding overlaps in new fine-tuning cannot
  undo possible prior exposure in the released weights. Report all outcomes, not only improvements.

## 2026-11-02 UTC — Bounded follow-up diagnostics, then stop

- Jack questioned whether more experiments justify compute. He authorized existing-artifact analysis
  followed by only two inference-only variants: original-BN-buffer restoration on the saved stage4
  model, and one fixed three-scale crop ensemble. No new training or unrestricted tuning authorized.
- Prespecified crop factors 0.9/1.0/1.1 at fixed288x384, flip retained. Aligned heatmaps using the
  actual repo affine transforms (width-1 convention), coverage-normalized uniform fusion before
  the existing decoder. No scale sweep. BN experiment restores running_mean/running_var/counter
  only; retains all trained parameters, including BN affine weights.
- Added unpublished tools/{bounded_diagnostics.py,analyze_bounded_diagnostics.py,
  run_bounded_diagnostics.py,analyze_bounded_diagnostics.sbatch,run_bounded_diagnostics.sbatch}
  and tests/test_bounded_diagnostics.py. Local nine geometry/BN tests passed; Linux CPU tests9/9.
  Full local suite26 tests passed with one GNU-timeout skip. No package installation/build.
- CPU `11556695` COMPLETED0:0, 17s script/allocation, d1027,2CPU/4G. Fresh home quota0.02TiB used
  /0.07TiB limit,43,514 inodes (rounded). Original/trained checkpoints safely loaded and hashed.
  Saved predictions analyzed without new inference. Per-person OKS is not confidence-ranked AP.
- Head-only per-person OKS correlation to original0.999928; all100 images within0.01OKS.
  Stage4 correlation0.977266;5 images improved>=0.01,6 worsened>=0.01,89 near. BN393 buffers
  changed across131 modules; some channel means shifted up to2.77 original std. Drift alone did
  not prove causation; the GPU buffer-only intervention tested its conditional contribution.
- GPU `11556720` COMPLETED0:0, out of queue, V100-SXM2-32GB/d1022,2CPU/8G, requested10m cap
  with latest-start deadline. Actual allocation47s, script45s. One allocation, no retries/Hopper,
  no new training. Matched original93.6071706887983 and trained93.198637527681 controls exact.
- BN-restored AP93.35239311605797, +0.153755588376967 vs trained but -0.25477757274033763 vs
  original. AR94.5. This supports normalization contributing to the observed degradation, not an
  assertion that BN was the sole cause or that training with frozen BN will improve the baseline.
- Fixed crop ensemble AP93.32060253033816, delta -0.28656815846014183 vs original, AR94.5.
  It used600 forward-image views versus200; CNN time3.934s vs1.351s. Single-pass wall9.478s
  vs3.682s is not a randomized app benchmark. This preset costs more and did not improve AP.
- Original and trained checkpoint files unchanged. Preserved diagnostic BN-restored raw state at
  output/bounded-diagnostics-11556720/BN_buffers_restored.pth, SHA256
  fe0942279648bc817e450823bbf37adad4bae0fe3da7998e6cbd7b2787b664e8 (Explorer home only).
- Reports/protocols/predictions via xfer to local output/bounded-diagnostics-11556720/;
  human summary summary.md; CPU report/stdout/accounting evidence in
  log/bounded-diagnostics-evidence-2026-11-02/. Analysis artifact on Explorer:
  output/bounded-analysis-11556695.json. Runtime source hashes retained in logs/manifest.
- Both hypotheses finished below original; stop at the agreed gate. No automatic follow-up
  fine-tuning, scale search, dataset acquisition, source data changes, cancellation, commit or push.
  These are exploratory follow-ups on a previously inspected small benchmark, not independent
  proof of generalization or statistical significance. Original remains best of tested variants.

## 2026-11-05 — Reproduction tooling and final result selection

- Added the ordinary course-required AGENT.md entry point linking to canonical AGENTS.md.
  Selected the original checkpoint in MANIFEST.md with actual run IDs, source-revision caveat,
  config/data/model hashes, metrics and evaluation command. Corrected stale setup/training status.
- Added safe asset preparation and offline/optional-COCO result checks under reproducibility/.
  Original datasets and weights remain unchanged and outside Git.
- Added diagnostic path/control arguments for new screen IDs: CPU analysis accepts
  --screen-dir/--trained-sha256/--expected-trained-ap and corresponding FIDIP_DIAG_* sbatch
  variables. GPU diagnostics use those pinned analysis fields. Historical defaults and metric/
  inference math are unchanged. Local tests cover argument parsing and checkout-root boundaries;
  this interface was not separately rerun on a GPU.
- Tests cover safe extraction and overwrite/traversal/symlink/case-collision refusal, preservation
  of existing data and the diagnostic argument guards. Recorded AP/final-step/freeze/control
  consistency checks pass. The separate Mac environment lacks pycocotools, so optional six-way
  COCO rescoring was not rerun or installed here; original Explorer evaluation evidence is retained.
- No new pose-model training/inference, environment installation, Explorer access, data transfer,
  SLURM job, cancellation, Git commit or push occurred during this cleanup.

## 2026-11-05 — Lean publication scope

- Jack approved preparing a lean publication of the actual ML source, tests/configs, required
  documentation and recorded results, rather than every intermediate artifact.
- Only the top-level FiDIP_Report.pdf is eligible for report-related publication. The report/
  directory and local packaging/review artifacts are ignored and not to be force-added.
- Documentation focuses on environment/data preparation, supplied model/code, baseline evaluation,
  controlled experiments, metrics, limitations and AI-assisted ML work.
- Small unchanged copies of the three main result JSONs are in reproducibility/results/ so offline
  score checks do not depend on ignored output folders or document assets. Original prediction
  files and full audit logs remain local; these are not supplied in a fresh Git clone.
- A copy of the Git-visible ML source and recorded results passed the standalone consistency
  check and all34 unit tests (33 passed/1 Mac GNU-timeout skip) without datasets, weights or a GPU.
  This is an offline source check, not a new Linux/CUDA end-to-end evaluation.
- Jack subsequently gave the go-ahead to commit and push this reviewed set to his fork. Preserve
  the local/remote working runtimes and inspect differences before any future synchronization.

## 2026-11-06 — Authorized lean source release

- Jack explicitly authorized pushing the reviewed 46-path set to his fork. Fetched origin and
  verified local master and origin/master matched at bc66664 before making publication commits.
- Runtime commit 6aa9e7b38f31a6ee3bda85711c2aab045f8aebd5 contains39 named paths: ML fixes,
  setup/evaluation/experiment helpers, tests, three unchanged metric JSONs and ignore rules.
- Required scientific documentation and the top-level final PDF are prepared as a separate7-path
  commit. No datasets, weights, raw prediction/audit outputs, compiled binaries or local authoring
  assets are added. Historical entries retain the source/publication state at their original dates.
- Pre-publication checks:34 unit tests (33 pass/1 Mac GNU-timeout skip), recorded-result consistency,
  Python/JSON/shell checks, exact staged-path/credential-pattern checks and whitespace checks pass.
  No new model runs or Explorer operations; its checkout is not automatically pulled or overwritten.
