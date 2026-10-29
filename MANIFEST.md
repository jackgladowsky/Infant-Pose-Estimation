# FiDIP Reproduction Manifest

This file inventories every input and output needed to reproduce reported results. Update paths,
checksums, versions, and run IDs whenever an artifact is obtained or produced.

**Documentation-only publication (2026-10-28):** the verified runs used compatibility fixes,
modified evaluation code, and cluster/validation helpers retained in the Mac/Explorer working
checkouts, intentionally not committed or pushed. References below inventory those working
artifacts; they do not imply every script, report, or payload is available in a fresh GitHub clone.
Runtime hashes identify the actual run inputs. Datasets, weights, binaries, and raw logs are excluded.

## Source repository

| Item | Value |
|---|---|
| Local checkout | `/Users/jackg/neu/eece7373/Infant-Pose-Estimation` |
| Fork remote | `git@github.com:jackgladowsky/Infant-Pose-Estimation.git` |
| Official repository | `https://github.com/ostadabbas/Infant-Pose-Estimation.git` |
| Starting branch | `master` |
| Starting commit | `a76b4f4` |
| License | Non-commercial use; see `README.md` |

## Explorer working directory

- Fork cloned at `/home/gladowsky.j/Infant-Pose-Estimation`, starting commit
  `a76b4f418107b8d3548d9779e94702e2508fbef1`.
- Remote `origin`: `https://github.com/jackgladowsky/Infant-Pose-Estimation.git`.
- Remote `upstream`: official repository above.
- Empty directories created: `data/`, `models/`, `output/`, `log/`, `slurm-logs/`.
- Jack confirmed personal home storage (75 GB limit) after the policy review: retain the existing
  checkout and repository-local `.venv`. First inspect actual quota/environment in a short CPU job;
  do not reinstall, move environments, or delete caches automatically. All home tools, managed
  Python/cache, data, logs, and checkpoints count toward the same quota.
  Dataset files were copied through the required transfer node on 2026-10-28 and CPU-validated
  by `11449533` (COMPLETED `0:0`). Test100 assets are ready; training issues remain unchanged.
  The original HRNet checkpoint is now present and verified; baseline job `11450265` reproduced
  93.6 AP (93.6071706887983 unrounded percent). No model training has run.
  `/scratch` is for running jobs only; persist retained outputs to home before a job ends.
- Setup/docs and the retrieved lockfile were committed and pushed to Jack's fork at `8f4c497`,
  then fast-forwarded on Explorer on 2026-10-22. The remote checkout was clean after synchronization.
- Original remote uncommitted setup is preserved in Git stash entry
  `FiDIP setup preserved before fork sync 8f4c497` (`stash@{0}` when created); do not drop it automatically.
- `/projects/aclab` is writable but Jack explicitly rejected it. Use `/home/gladowsky.j`, not any
  lab/group allocation; do not create/request a project allocation without approval.
- `/courses/EECE7398.202710` exists but this account has no read/write/traverse access.
- CPU inspection job `11305792` reported quota 0.01 TiB used / 0.07 TiB limit (rounded),
  33,220 inodes used. CPU/V100 native checks and original pretrained HRNet Test100 evaluation
  subsequently passed; see baseline run `11450265` below.

## uv setup (CPU/V100 smoke validation passed)

- `pyproject.toml`: Linux x86-64 training/evaluation dependencies, explicit CUDA 12.1 Torch index.
- `.python-version`: Python 3.12; installed 3.12.14 verified by CPU inspection.
- `.venv/`: CPU dependency/import checks and V100 native/CUDA/NMS/project/CLI smoke validation
  passed (`11308338`, `11308377`), followed by dataset validation and successful pretrained
  Test100 evaluation (`11450265`). Training remains unverified/not started.
- `uv.lock`: retrieved locally from Explorer on 2026-10-22 (41,582 bytes), with matching SHA-256
  `a9d18d4df5ab965229e9205e488b8b9938a773f3326db7c4bf281ad29dbfac4a`.
  Local/remote `pyproject.toml` SHA-256 also matches:
  `44900cbb39e036cb463ee46591230a4fa83cc3a4e7846cbb5032817f0a783dbb`.
- `ENVIRONMENT.md`: setup, stage timeouts/timings, and the CPU-build → V100-validation procedure.
- Reviewed revisions synced with matching SHA-256 checksums and CPU/V100 smoke-tested: `lib/nms/cpu_nms.pyx` integer dtype fix;
  `gpu_nms.pyx` scalar type/empty-input guard; `setup_linux.py` configurable V100+PTX build.
- New `tools/build_environment.sbatch`: CPU-only harness/native build/regression, 5-minute cap;
  first run `11308157` passed 10 harness tests then failed opening generated `cpu_nms.c` during
  Cython compilation (build stage 3s, process total 5s). Log `slurm-logs/build-11308157.out`.
  CPU-only retest `11308338` uses `--cython-c-in-temp`; log `slurm-logs/build-11308338.out`.
  Script PASS: harness 10/10, compiled CPU-NMS cases 6/6; native compilation 12s, script total
  15s. Accounting COMPLETED `0:0`, allocation elapsed 4m56s. GPU job `11308377` subsequently
  COMPLETED `0:0`, allocation 2m58s, script 26s; log `slurm-logs/env-11308377.out`.
  Actual GPU: Tesla V100-SXM2-32GB on `d1007`, capability 7.0, driver 545.23.08. All 7 Python
  checks, 6 CPU/6 GPU NMS fixtures, and both CLI import checks passed. Job is out of the queue.
  Build-script SHA-256 `3ed295441b8653d6d6e11df84ea11af4ab052d3a9bc37870a64b911fd71eb3cc`;
  setup SHA-256 `8cb5c9b9f1271eb180fc8774424bd1aacc198b7984c8ceef2770ff7281ddc704`.
  Uses uncommitted reviewed source over `7bc14bc`; full source hashes are emitted by each job.
- Revised `tools/check_environment.sbatch`: V100-only runtime/import checks, 5-minute cap, no build.
- New `tools/environment_check_common.sh`: UTC stage timings, exit codes, bounded commands.
- Revised `tools/check_environment.py`: phased/timed checks, stall stack traces, six NMS fixtures,
  and a V100/A100 model guard. No changes to dependency pins or training configuration.
- `tests/test_environment_checks.py`: local harness/unit regressions; not a native/GPU validation.
- Original `7bc14bc` GPU-check scripts ran as job `11305949` on H200 node `d4055` (untyped
  resource-request mistake). Final state TIMEOUT, reported elapsed 10m30s; no queued entry remains.
  Log: `slurm-logs/env-11305949.out`. Native builds, train/eval imports, and CUDA matmul passed;
  CPU NMS raised an integer buffer dtype mismatch. GPU NMS was not tested. Local future batch
  script requests V100-SXM2 explicitly; successful replacement `11308377` is recorded above. Generated native files
  on Explorer differ from Git and must not be published as source changes.
- `tools/inspect_environment.sbatch`: CPU-only quota/installed-environment inspection; copied to
  Explorer and submitted as job `11305792` on `d0023`. Base `7bc14bc` plus uncommitted script,
  SHA-256 `046a940dc860ac799139e2236ecc39640feaeb3640e0142012d2b0cc477f1be6`.
  Output: `slurm-logs/inspect-11305792.out`; final PASS for quota reporting, CPU imports/matmul,
  and all 51 installed-package dependency checks. Accounting confirms COMPLETED, exit `0:0`,
  elapsed 4m44s; controller cleanup was still observed afterward.
- uv cache: `/home/gladowsky.j/.cache/uv`; managed Python resolved to
  `/home/gladowsky.j/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12`.
- `log/uv-sync.log`: installation log on Explorer; lists installed PyTorch 2.3.1+cu121 and other
  dependencies. No active sync process observed during the 2026-10-22 review.

## Project context and course artifacts

- `AGENTS.md`: canonical agent context and chronological research notebook.
- `AGENT.md`: former compatibility symlink, removed at Jack's request on 2026-10-28. Use
  `AGENTS.md`; review the course's singular-name requirement again for final submission.
- `SKILL.md`: project-specific Explorer operations and safety/pre-training checks.
- `docs/explorer/SKILL.md`: unchanged reference copy of the sibling course repo's `SKILL.md`,
  added 2026-10-22; related course notebooks/recipes are not vendored. Use the root guide for
  actual FiDIP paths, configuration, and safety rules.
- `../STUDENT_GUIDE.md`: local assignment requirements; outside this repository, not assumed
  available on Explorer. Core requirements are summarized in `AGENTS.md`.
- `../EECE-7373-Machine-Learning-with-Small-Data/`: local course Explorer tutorials/templates;
  not the FiDIP environment specification.

## Environment

| Artifact | Purpose | Status |
|---|---|---|
| `fidip_env.yml` | Conda environment (`fidip`, Python 3.12.4, PyTorch 2.3.1, CUDA 12.1) | Present; not yet validated on Explorer |
| `lib/Makefile` | Build Linux CPU/GPU NMS extensions | Present; not yet run |
| `.env.example` | Documents optional secret names without values | Present |
| `.env` | Local secrets | Not created; always gitignored and never inventoried by value |

## Primary configuration and entry points

| Path | Purpose | Status |
|---|---|---|
| `experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml` | HRNet-W48 FiDIP train/eval config | Present |
| `tools/train_adaptive_model_hrnet.py` | HRNet FiDIP training | Present |
| `tools/test_adaptive_model.py` | FiDIP evaluation | Present |
| `tools/train_adaptive_model.py` | ResNet FiDIP training | Present |
| `tools/train_adaptive_model_mobile.py` | MobileNet FiDIP training | Present |
| `syn_generation/` | Synthetic infant generation pipeline | Present; not yet used |

## Datasets

Datasets must not be committed. Jack chose the existing personal home checkout, with a 75 GB
whole-home quota. Do not use lab/project storage or create allocations without approval.

| Dataset | Expected repository path | Actual storage path | Access/status | Checksum/version |
|---|---|---|---|---|
| SyRIP annotations | `data/syrip/annotations/` | Local relative symlinks into `data/raw/` | Three expected JSONs present; Test100 baseline ready; training review required | Per-file SHA-256 in `data/validation-report.json` |
| SyRIP infant train images | `data/syrip/images/train_infant/` | Local `data/raw/fidip/syrip/images/train_infant/` | 1,500 files, 1,200 referenced; 46 dimension mismatches, 4 Test100 duplicates | `data/image-manifest.json` |
| SyRIP pre-infant train images | `data/syrip/images/train_pre_infant/` | Local `data/raw/fidip/syrip/images/train_pre_infant/` | 4,057 files/records; existing loader filters to 3,904 usable | `data/image-manifest.json` |
| SyRIP validation images | `data/syrip/images/validate_infant/` | Local `data/raw/original/SyRIP/images/validate_infant/` | 500 available files; working original Test100 uses 100, all checks pass | `data/image-manifest.json` |

### Explorer copy status (2026-10-28; transfer/Test100 validated)

- Required transfer endpoint: `gladowsky.j@xfer.discovery.neu.edu` (`xfer-00`), not login/compute
  nodes. [Official RC transfer policy](https://rc-docs.northeastern.edu/en/latest/datamanagement/transferringdata.html).
  Passwordless SSH and strict host-key verification succeeded using the existing key.
- Destination: `/home/gladowsky.j/Infant-Pose-Estimation/data/`; shared home path is visible on
  both xfer and Explorer. No `/projects` or lab allocation was used.
- Quota preflight `11448779`: COMPLETED `0:0`, 0.01 TiB used / 0.07 TiB limit (rounded),
  35,720 inodes used; log `slurm-logs/data-quota-11448779.out`.
- Rsync `-rltp --partial --timeout=120 --bwlimit=20000 --stats`, no --delete, exited 0 via xfer
  after 273s. Source tree: 1,948,049,676 bytes / 7,705 entries. Raw trees, ZIPs, relative links,
  and local audit snapshots were copied. CPU validation `11449533` confirmed matching archive,
  image, and annotation hashes; no new transfer/image/schema problems were introduced.
- Receiver log: `log/data-xfer-rsync-20261028.log`. Login-endpoint attempts before discovering
  the requirement were unsuccessful; never repeat that transfer route. One known partial ZIP
  receiver temporary (about 403 MiB) remains outside the working profile; final ZIP hashes are
  now verified, but no cleanup/deletion was performed during validation.
- CPU validation `11449533` on `d0005`: COMPLETED `0:0`, allocation elapsed 31s, script elapsed
  28s (transfer checker 27.663s). It verified both ZIP sizes/hashes, fully decoded/rehashed all
  **7,657** images, compared every source and working annotation hash, and matched the known
  training warnings/errors to the Mac audit. Test100 passes; training is still blocked for review.
- Input identity hashes were checked before submission and printed by the job. No GPU, training,
  label fixes, environment installs, or new downloads were involved.
- Remote evidence: `slurm-logs/data-11449533.out`, `data/explorer-transfer-validation.json`,
  refreshed `data/validation-report.json` and `data/image-manifest.json`; immutable `*.local.json`
  snapshots remain for comparison. Logs/summary were copied back through xfer to
  `data/explorer-validation/2026-10-28/job-11449533.out` and `job-11449533-summary.json`.
- Current flags: archive_hashes_match=true, image_hashes_and_dimensions_matched=7657,
  annotation_hashes_match=true, baseline_test100_ready=true, training_ready_without_review=false.
  No dataset-validation job remains queued/running.
- Tool SHA-256: batch `c1a9ac2b05e1183925acf01a426543f56bf643e1fdea2e134d60923a5f69fb9b`;
  Python `2ce8fea83e0b1c3a71558412e685508e0d28e6dbc5e45b797d578b4ab765b12d`.

### Local acquisition and validation (2026-10-28)

Jack authorized moving both ZIPs from `~/Downloads` into this repository and extracting/checking
all assets. Paths in this section resolve under the Mac checkout, not Explorer. `/data/` is
Git-ignored; no images/archives/annotations are staged or committed. No source code or supplied
annotation/image was modified, and nothing was transferred to Explorer.

| Archive | Retained path | Extracted root | SHA-256 |
|---|---|---|---|
| Original SyRIP | `data/archives/SyRIP.zip` | `data/raw/original/SyRIP/` | `1d13e816375988be9bbdab8bf5be6562a5b9bfa83684d57ccbfec9dd619e1692` |
| FiDIP training | `data/archives/syrip_for_train.zip` | `data/raw/fidip/syrip/` | `1f267878732fb81776c5c7f153bad45d652cadf5bfd83761971390b64d6cee1b` |

Archive sizes match the earlier source metadata; all members extracted with CRC validation.
All **7,657** image files fully decoded and were SHA-256 hashed; no corrupt/unreadable or missing
referenced images, loader filename mismatches, or EXIF-orientation problems were found.
Total local data footprint is about **1.81 GiB**, including retained archives and raw extractions.

Working relative-symlink profile:

- `annotations/person_keypoints_train_pre_infant.json` and `images/train_pre_infant` → FiDIP raw tree.
  4,057 annotation/image records; the existing loader excludes 153 invalid boxes, leaving **3,904**
  (1,876 real / 2,028 synthetic). This filtering is not a failed extraction or an automatic label edit.
- `annotations/person_keypoints_train_infant.json` and `images/train_infant` → FiDIP raw tree.
  **1,200** referenced samples (200 real / 1,000 synthetic); 300 extra image files are not selected.
- `annotations/person_keypoints_validate_infant.json` → original `annotations/validate100/` JSON;
  `images/validate_infant` → original validation images. **100** samples, all baseline checks pass.
  Original Test500 and alternate original train annotations remain intact in the raw tree.
- Transfer the complete `data/` tree preserving relative symlinks, or deliberately materialize
  the working profile; copying only links will omit their targets.

**Readiness:** `baseline_test100_ready=true`; `training_ready_without_review=false`.
Working Test100 has valid 17-joint annotations, correct image dimensions, and 100 loader-usable
records. Pretrained HRNet evaluation has now run successfully as `11450265` (see results below).
This does not resolve or authorize training on the flagged inputs.

Training caveats (preserved, not silently fixed):

1. FiDIP training bundle has **46 JPEG/annotation dimension mismatches**. For example,
   `train10000.jpg` is 1920×1080 but its annotation declares 640×480.
2. **Four byte-identical training/Test100 pairs occur in both released train variants**:
   `train00272.jpg`/`test46.jpg`, `train00073.jpg`/`test20.jpg`,
   `train00315.jpg`/`test42.jpg`, `train00340.jpg`/`test50.jpg`.
   No pre-infant/Test100 byte-identical overlap was found. This checks exact bytes, not near-duplicates.
3. Original and processed train variants reference the same image-payload set, but synthetic
   keypoint/bbox annotations differ; do not silently replace one training protocol with the other.
   The original pairing has no dimension mismatches, but retains the same four holdout duplicates.
4. Full reports preserve `num_keypoints` metadata and out-of-frame coordinate warnings. The
   training-bundle validation `test11.jpg` also has a size mismatch; the working profile avoids
   it by using the consistent original validation image folder. Test100 annotation IDs/core labels
   are identical between the two ZIPs.

Reports/recheck:

- `data/README.md`, `data/archive-manifest.json`, `data/image-manifest.json`, `data/validation-report.json`.
- `../.venv/bin/python -B data/validate_datasets.py` from this Mac checkout. Exit **1** deliberately
  flags training caveats; it does not mean Test100 or extraction failed. On Explorer use its own
  `.venv/bin/python` in a CPU allocation after transfer.
- Working annotation SHA-256: pre-infant `cbad22e2f116d912b43e225f9833efe832f8a05d136ea5ee7ba5a9383f8463b2`;
  train `003aaccac1698e48251f0b76d335c2369eefb40610def9c5cc359e661b8e0505`;
  Test100 `8b299a879a95913483498e3f2115a826ac23ebc5262c3efb8d65f63718e3194c`.

### Download/access verification (2026-10-22; historical metadata-only check)

- Original archive: `https://coe.northeastern.edu/Research/AClab/SyRIP/SyRIP.zip`, 327,912,103 bytes.
  Complete ZIP central directory: 2,006 files, none encrypted, total unpacked size about 0.31 GiB.
  Annotation paths include both `validate100/` and `validate500/`; use the original Test100
  annotation explicitly for the published 93.6 AP comparison.
- FiDIP training archive: `https://coe.northeastern.edu/Research/AClab/SyRIP/syrip_for_train.zip`,
  637,172,858 bytes. Complete central directory: 5,660 files, none encrypted, unpacked about
  0.60 GiB. Root `syrip/annotations/` contains all three expected annotation filenames below.
- Expanded archive: `https://coe.northeastern.edu/Research/AClab/Expanded_SyRIP/expanded_SyRIP.zip`;
  sampled 500 file entries, all encrypted. Obtain an authorized password from the authors/instructor
  only if that variant is needed. Do not conflate expanded-data results with the original baseline.
- Used bounded HTTP range reads of ZIP metadata only; no full archive or image payload acquired,
  no checksum yet, and no Explorer download job submitted. Earlier blanket restricted-access notes
  are superseded for the original/training ZIPs. Verify split counts/schema/checksums after download.

Expected annotation files:

- `person_keypoints_train_pre_infant.json`
- `person_keypoints_train_infant.json`
- `person_keypoints_validate_infant.json`

## Downloaded model inputs

Model files must not be committed.

| Model | Expected repository path | Source/status | SHA-256 |
|---|---|---|---|
| Pretrained FiDIP HRNet | `models/hrnet_fidip.pth` | User-supplied original-folder download; present on Mac and Explorer, CPU strict-load/forward verified, baseline reproduced | `bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07` |
| Pretrained FiDIP MobileNet | `models/mobile_fidip.pth` | Missing; FiDIP Google Drive folder linked from `README.md` | TODO |
| DarkPose HRNet-W48 | `models/coco/w48_384x288.pth` | Missing; TGA Google Drive folder linked from `README.md` | TODO |

### HRNet checkpoint acquisition and verification (2026-10-28)

- Moved `~/Downloads/hrnet_fidip.pth` into local gitignored `models/` (255,354,649 bytes), retaining
  its bytes unchanged. Added `models/hrnet_fidip.metadata.json`; source is the original FiDIP
  Google Drive folder linked in README. No author-published checksum was available; provenance
  is the user-supplied download, not independently authenticated by filename alone.
- Local ZIP CRC passed. Safe `torch.load(..., map_location='cpu', weights_only=True)` returned
  1,754 tensor entries, with no missing/extra keys or shape mismatches for the configured HRNet.
- Copied model/metadata via xfer only into Explorer home `models/`; remote CPU admission job
  `11450222` COMPLETED `0:0` on `d1027`, 13s allocation / 12s script. It verified SHA-256,
  finite tensors, strict state-dict coverage, and a CPU forward: feature `[1,48,96,72]`,
  heatmap `[1,17,96,72]`. Report: `log/checkpoint-11450222.json`.
- Tightened `tools/test_adaptive_model.py` to weights-only CPU deserialization and `strict=True`,
  replacing unsafe/lenient loading. For this exact matching checkpoint no inference math changes.
- Added `tools/check_checkpoint.sbatch` (CPU-only) and `tools/evaluate_baseline.sbatch` (explicit
  V100 default); both capped at 5 minutes. No dependency installation or model training.
- Local verification artifacts: `models/validation/checkpoint-11450222.json`,
  `log/baseline-evidence-11450265/`; exact inputs/script/config hashes are in SLURM logs.
- Existing DarkPose adult and MobileNet weights are not needed for this pretrained HRNet evaluation.

## Reference baseline

The repository reports the following SyRIP Test100 HRNet-W48/DarkPose metrics:

| Method | AP | AP50 | AP75 | AR | AR50 | AR75 |
|---|---:|---:|---:|---:|---:|---:|
| DarkPose + fine-tune | 92.7 | 98.5 | 98.5 | 93.9 | 99.0 | 99.0 |
| DarkPose + FiDIP | **93.6** | 98.5 | 98.5 | 94.6 | 99.0 | 99.0 |

The FiDIP reference was reproduced on 2026-10-28 by `11450265`: **93.6071706887983 AP percent**,
which rounds to the published **93.6**. This is reproduction within reported rounding, not an
improvement claim. Full metric vector (percent): AP50/AP75 **98.52475247524751**, AR **94.6**,
AR50/AR75 **99.0**. COCO medium-area AP/AR are -1 because that size category has no applicable
samples; all/large metrics are valid.

Run settings: original Test100 (100 images, annotation SHA-256
`8b299a879a95913483498e3f2115a826ac23ebc5262c3efb8d65f63718e3194c`), GT boxes, flip testing,
postprocessing, 384×288 model input, batch 16, official YAML unchanged. Only output/log paths,
debug rendering (disabled), and log frequency were overridden; model/dataset math was unchanged.

GPU job `11450265` depended afterok on checkpoint check `11450222`. Actual GPU was V100-SXM2-32GB,
node `d1020`, driver 545.23.08. COMPLETED `0:0`, 71s allocation / 64s script; inference/evaluation
stage 60s, independent full-precision COCO rescoring 1s. No active baseline job remains.

Artifacts on Explorer:

- `output/baseline-test100-11450265/metrics.json` (full precision and provenance).
- `output/baseline-test100-11450265/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant/results/keypoints_validate_infant_results_0.json`.
- Same model/config output folder contains upstream `feature.npy` and evaluation log.
- `slurm-logs/baseline-11450265.out`, `slurm-logs/checkpoint-11450222.out`.
- Local xfer-retrieved copies: `output/baseline-test100-11450265/{metrics.json,predictions.json}`,
  `models/validation/checkpoint-11450222.json`, `log/baseline-evidence-11450265/`.

Source is still uncommitted reviewed changes over `7bc14bc`, not a newly pushed revision. Key
SHA-256: YAML `cd1151abad97aac80400252c89e000c3ea7fca87975f87b06546e66cc1191157`;
entry point `7c0902b27c9985671dbad0c31b8755ea2c270621455adb74e2635ca6c8b4a113`;
baseline script `182fba6d0b1f648a07f4b67d062576b74aff9479e09027a303f46d7609406048`;
CPU-check script `b75d197869d8930cf07f90725157575c29243ebfb09ecc712f06e65960cd22e8`.

Baseline evaluation command:

```bash
python tools/test_adaptive_model.py \
  --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
  TEST.MODEL_FILE models/hrnet_fidip.pth TEST.USE_GT_BBOX True
```

## Produced experiments and checkpoints

| Run | Git commit | Config | Dataset version | Job/W&B ID | Metrics | Checkpoint/log path | Status |
|---|---|---|---|---|---|---|---|
| Baseline reproduction | `7bc14bc` + reviewed uncommitted changes; hashes above | HRNet config above | Original Test100, 100 images | CPU `11450222`, GPU `11450265` | AP 93.60717; AP50/AP75 98.52475; AR 94.6; AR50/AR75 99.0 | `output/baseline-test100-11450265/`, `models/hrnet_fidip.pth` | Reproduced published rounding; COMPLETED `0:0` |

## Final reported result

No final improvement run selected yet. The reproduced baseline above is the comparison reference.
Fill this section after controlled improvement experiments are reproduced and verified:

- Run ID: TODO
- Git commit: TODO
- Config: TODO
- Dataset version: TODO
- Checkpoint path and SHA-256: TODO
- AP/AP50/AP75/AR: TODO
- Evaluation command and log path: TODO
