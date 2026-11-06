# FiDIP Reproduction Manifest

This file inventories every input and output needed to reproduce reported results. Update paths,
checksums, versions, and run IDs whenever an artifact is obtained or produced.

**Documentation-only publication (2026-10-28):** the verified runs used compatibility fixes,
modified evaluation code, and cluster/validation helpers retained in the Mac/Explorer working
checkouts, intentionally not committed or pushed. References below inventory those working
artifacts; they do not imply every private dataset, checkpoint or audit output is in Git.
That earlier source-publication restriction is superseded by the user-authorized lean source release.

The lean source publication contains ML code, tests/configs, required documentation and the recorded
results under `reproducibility/results/`. Follow `reproducibility/README.md` for input acquisition
and evaluation commands. Runtime hashes identify actual run inputs; datasets, weights, compiled
binaries and full audit logs remain external.

## Source repository

| Item | Value |
|---|---|
| Local checkout | `/Users/jackg/neu/eece7373/Infant-Pose-Estimation` |
| Fork remote | `git@github.com:jackgladowsky/Infant-Pose-Estimation.git` |
| Official repository | `https://github.com/ostadabbas/Infant-Pose-Estimation.git` |
| Starting branch | `master` |
| Starting commit | `a76b4f4` |
| Runtime source release | `6aa9e7b38f31a6ee3bda85711c2aab045f8aebd5` (fixes, drivers, tests, recorded metrics) |
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
  93.6 AP (93.6071706887983 unrounded percent). A pose-only pilot and two fixed-budget fine-tunes
  subsequently completed; original adversarial training was not reproduced.
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
  Test100 evaluation (`11450265`), then the verified pilot and controlled 600-step continuations.
  This does not validate the upstream full adversarial trainer.
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
- `AGENT.md`: short ordinary Markdown course entry point added for final review (2026-11-05),
  linking to canonical `AGENTS.md`; the former symlink remains removed.
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
| `lib/Makefile` | Upstream native-build recipe | Reference only; validated build uses `tools/build_environment.sbatch` with temporary C generation |
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
| Pretrained FiDIP MobileNet | `models/mobile_fidip.pth` | Not used/not acquired; outside this HRNet study | Not applicable |
| DarkPose HRNet-W48 | `models/coco/w48_384x288.pth` | Not used/not acquired; released FiDIP weights initialize this study | Not applicable |

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

At the baseline run, source consisted of reviewed uncommitted changes over `7bc14bc`. Key
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
| Head-only continuation | At run time: local runtime over docs HEAD `bc66664`; runner hash below | 600 steps, batch8, Adam1e-5, seed7373, 833 trainable params | Approved filtered 1,150 train; original Test100 eval | GPU `11553889` / head | AP93.54721, delta -0.05996; AR94.5 | `output/small-screen-11553889/head/final_pose.pth` | COMPLETED `0:0`; no improvement in one-seed screen |
| Stage4 + head continuation | Same runtime source and initialization | Same budget/schedule; 45,894,881 trainable params | Same train selection / Test100 | GPU `11553889` / stage4_head | AP93.19864, delta -0.40853; AR94.3 | `output/small-screen-11553889/stage4_head/final_pose.pth` | COMPLETED `0:0`; lower AP in screen |
| Flip testing off | Original released weights, no training | Existing decoder/GT boxes, flip off only | Original Test100 | GPU `11553889` / original_flip_off | AP91.91587, delta -1.69130; AR93.4; ~2x CNN-forward speed only | `models/hrnet_fidip.pth`, run predictions/results | COMPLETED `0:0`; accuracy/compute tradeoff |
| BN-buffer restoration | Inference diagnostic; learned parameters unchanged | Original BN running buffers substituted into saved stage4/head model | Original Test100 | GPU `11556720` / BN_buffers_restored | AP93.35239; +0.15376 vs trained, -0.25478 vs original; AR94.5 | `output/bounded-diagnostics-11556720/BN_buffers_restored.pth` | COMPLETED `0:0`; partial recovery, not a new best |
| Fixed crop-scale TTA | Original released weights, no training | Crop factors0.9/1/1.1; flips; affine-aligned, coverage-normalized heatmap mean | Original Test100 | GPU `11556720` / three_crop_TTA | AP93.32060, delta -0.28657; AR94.5; 3x image views | Original weights plus diagnostic runner/protocol | COMPLETED `0:0`; more compute, lower AP; no sweep |

### Fine-tuning readiness and diagnostic pilot (2026-11-01)

- Helpers introduced for this run (now included in the source release): `tools/audit_finetune_readiness.py`, `tools/audit_finetune.sbatch`,
  `tools/pilot_finetune.py`, `tools/pilot_finetune.sbatch`, `tests/test_finetune_pilot.py`.
- CPU audit `11553238`: COMPLETED `0:0`, `d1027`, 25s script / 92s allocation. Three CPU resume
  unit tests passed; source checkpoint and data payload/annotation hashes verified unchanged.
- Reconfirmed 46 synthetic dimension mismatches and 4 real train/Test100 byte duplicates. Proposed
  unflagged comparison pool is 1,150 records (196 real, 954 synthetic), NOT yet an approved split.
  Original releases/working labels unchanged; no samples removed from the datasets.
- Corrected planning assumption: author's `BalancedBatchSampler` already produces 1:1 real/synthetic
  batches. A future 'more real' comparison must differ from that effective baseline, not pretend
  the 200/1,000 image inventory was the author's sampling ratio. No five-run comparison started.
- Admission JSON: `output/finetune-readiness-11553238.json`; audit stdout:
  `slurm-logs/train-audit-11553238.out`; local xfer copies: `log/finetune-evidence-2026-11-01/`.
- Diagnostic pilot `11553500`: gpu-short, typed V100-SXM2, 4 CPUs, 8G host RAM, 5m limit;
  start deadline used to avoid a long queue. Started ~34s after submission on d1019, V100-SXM2-32GB.
  H200 forecasts were long so none submitted. Pilot COMPLETED `0:0`, out of queue; 135s
  allocation, 56s script, 44.277s Python body. No scientific AP result or full-data comparison yet.
- Pilot uses only 32 explicitly verified examples (16 real/16 synthetic), batch 8, 50 optimizer
  updates at Adam LR 1e-5 plus two 3-step branches for resume equivalence. Full pose model, FP32,
  deterministic/TF32 off, supervised pose MSE only; original augmentation geometry retained.
  This is not full-data training or adversarial FiDIP reproduction. No Test100 scoring/selection.
- Output folder: `output/finetune-pilot-11553500/` contains `protocol.json`, `pilot-report.json`,
  and `resume.pth` (764,835,130 bytes, step 50, pose+Adam+RNG; NOT a raw pose-only final model).
  Finite loss/gradients, head updates, exact 3-step uninterrupted/restored pose/Adam/RNG equivalence,
  and original checkpoint identity all verified. Original `models/hrnet_fidip.pth` remains immutable.
- Steady median step 0.3895523s, GPU peak allocated 4.299 GiB / reserved 4.990 GiB, batch MaxRSS
  1,492,904 KiB. Compute-only estimate 500 steps 3.246m / 1,000 steps 6.493m. These tiny-subset
  timings include exhaustive gradient checks and exclude full-data I/O/evaluation/queue overhead.
- Reports/protocol retrieved via xfer to local `output/finetune-pilot-11553500/`; checkpoint stays
  in Explorer home only. CPU/GPU stdout and accounting copied to `log/finetune-evidence-2026-11-01/`.
  All outputs gitignored. Both jobs completed; no active audit/pilot remains.
- Training exclusions/repairs and independent development-validation protocol still require an
  explicit choice. Filtering future training cannot undo possible prior Test100 exposure in the
  released weights; a training-derived validation subset may also have been seen during pretraining.

### Completed three-test screen (2026-11-01)

User selected a smaller screen instead of expanded-data acquisition or the earlier five runs:
head-only fine-tuning, final-stage+head fine-tuning, and original-checkpoint flip-off inference.
The separate exclusion manifest is now approved for these trials: 1,150 retained records
(196 real/954 synthetic), original data/labels and Test100 untouched. No manual label repairs.

Both fine-tunes: original FiDIP initialization, 600 steps, batch8, balanced 1:1 replacement
sampling, LR1e-5/Adam, seed7373, existing geometry augmentation, frozen early BatchNorm stats.
Same step-indexed image/augmentation schedule. Final Test100 evaluation only, no checkpoint
selection by Test100. Flip-off uses original weights; matched flip-on control must first reproduce
93.60717 AP. Timing labels distinguish GPU CNN-forward measurements from full evaluation walltime.

Sources introduced for this run: `tools/{prepare_small_screen.py,prepare_small_screen.sbatch,run_small_screen.py,run_small_screen.sbatch}`,
`tests/test_small_screen.py`; local screen tests 4/4 and resume tests 3/3 passed.
CPU preparation `11553804`: seven Linux unit tests and fresh payload/hash validation passed;
script rc0/18s, but allocation later TIMEOUT 0:0 / 5m08s after remaining RUNNING despite successful
script exit. Cause unverified; retained logs. Approved manifest `output/small-screen-profile-11553804.json`;
log `slurm-logs/screen-prep-11553804.out`; both retrieved via xfer to `log/small-screen-evidence-2026-11-01/`.

GPU `11553889` admitted from the verified complete CPU outputs without depending on its anomalous
Slurm lifecycle; runtime independently hashes all pinned inputs. gpu-short, typed V100-SXM2,
4CPU/8G/15m, latest-start deadline now+17m, d1022, V100-SXM2-32GB. COMPLETED `0:0`, out of queue:
6m45s allocation, 6m18s script, 6m16s experiment stage. All three tests finished; original flip-on
control exactly rechecked 93.6071706887983 AP. MaxRSS 3,476,404 KiB.
Outputs: `output/small-screen-11553889/{protocol.json,results.json}`, per-variant predictions and
300/600-step resume/final pose checkpoints; stdout `slurm-logs/small-screen-11553889.out`.
Full-precision AP: head-only **93.5472078062278** (delta **-0.05996288257050253**), stage4+head
**93.198637527681** (delta **-0.40853316111730464**), original flip-off **91.9158670222938**
(delta **-1.6913036665045098**). AR94.5/94.3/93.4 respectively. Both fine-tunes verified 600 updates,
head-weight changes, and unchanged frozen parameters/BN buffers. No statistical significance
established from this single-seed screen. Original remains the best tested checkpoint.

Paired warmed batch16 CNN-only median latency: flip on 13.00666 ms/original image vs off
6.50483 ms/image, **1.99954x forward-only speedup**. Nine alternating-order repeats; excludes
I/O, preprocessing, transfers, flip-back and decoding. Do not call it 2x application throughput.
Single-pass eval wall times 5.779s/3.516s are cache/order affected, not a controlled app benchmark.

Checkpoint SHA-256:
- Head: `9a8f03f65328baceb05ce36a111d9d7b71337d182e4a59dbfa241ee4e5a667ff`.
- Stage4/head: `b146f8c25d5c45f882603f858d1c532d054e1caedc687fedb41e154fd3be3f6e`.
- Selection profile: `42c193dc87f62eb0ba1aba60544bd95e4a4575d651c80c0320a05fde9cf6edc7`.
- Runner: `3c44a515adbef57b2f913ab235ad27096c430eded53a705f7fa6227b92f0e722`.

Human summary `output/small-screen-11553889/summary.md`; complete results/protocols/predictions
retrieved via xfer to the same local run folder. Prep/profile/stdout/accounting evidence under
`log/small-screen-evidence-2026-11-01/`. Large checkpoints remain Explorer home only. New final
pose checkpoints are raw state dicts; 300/600-step resume files also contain optimizer/RNG/protocol
and must not be passed directly to the original evaluation CLI.

No active jobs remain. No expanded datasets, new source/data publication, original-label edits,
raw-data deletions, new Git commits/pushes, or queued-job cancellations. Do not launch more training
merely because these experiments did not improve the baseline; agree on a new hypothesis first.

### Bounded follow-up diagnostics (2026-11-02 UTC)

User authorized existing-artifact analysis and only two additional inference hypotheses, with a
10–15 GPU-minute ceiling; no training. One 10m-capped allocation actually used **47 seconds**.

CPU `11556695` COMPLETED0:0,17s: nine Linux geometry/BN tests passed; original/trained checkpoint
hashes checked; existing saved predictions analyzed. Head-only per-person OKS correlation to
original0.999928, all100 images within0.01OKS. Stage4 correlation0.977266:5 improved>=0.01,
6 worsened>=0.01,89 within0.01. Per-person OKS/joint distance is not confidence-ranked COCO AP.
BN393 buffers changed in131 modules. CPU report `output/bounded-analysis-11556695.json`.

GPU `11556720` COMPLETED0:0,47s allocation/45s script, V100-SXM2-32GB/d1022,2CPU/8G.
Original and trained controls exactly reproduced93.6071706887983 and93.198637527681 AP.
- BN-only restoration: **93.35239311605797 AP**, +0.153755588376967 vs trained, but
  -0.25477757274033763 vs original. All learned parameters/BN affine weights retained.
  Supports a conditional normalization contribution, not that all degradation is caused by BN
  or that freezing BN during training will necessarily beat the released model.
- Fixed crop ensemble: **93.32060253033816 AP**, -0.28656815846014183 vs original.
  Crop extents0.9/1/1.1 at unchanged288x384 input; aligned via actual repo(width-1) affine geometry,
  coverage-normalized uniform heatmap fusion before original decoder; flip retained. No scale sweep.
  600 forward-image views vs200; CNN time3.934s vs1.351s. Single-pass wall9.478s vs3.682s is not a
  randomized app benchmark. This particular preset added cost and reduced accuracy.

Reports/protocols/predictions/human summary: `output/bounded-diagnostics-11556720/` on Mac/Explorer.
CPU/stdout/accounting xfer copies: `log/bounded-diagnostics-evidence-2026-11-02/`.
BN-restored raw state on Explorer only, SHA256
`fe0942279648bc817e450823bbf37adad4bae0fe3da7998e6cbd7b2787b664e8`.
Runner SHA256 `46f42d172983818e51aa4f84cec7504ebbca582c6d8f5920fdfd13f21d3ba0f3`;
helper `5a7ebf0181f4e25102119cd082f9727b0c6af3548642ed371c46b550d34bb107`;
CPU analysis artifact `deb88e796a55fdfb80ffb8dd2f443c1d21106b171805cbf27259056a3689fca9`.
Source weights/datasets unchanged. Both diagnostics finished below baseline. No active jobs,
new fine-tuning, retries, parameter sweeps or Git pushes. STOP at the agreed gate; further work
requires a new hypothesis/authorization. Findings are exploratory on previously inspected Test100.

### Recorded results and reproduction checks

- `reproducibility/results/baseline.json`: original checkpoint evaluation, job11450265.
- `reproducibility/results/small-screen.json`: matched controls, two fine-tunes and flip-off,
  job11553889.
- `reproducibility/results/bounded-diagnostics.json`: BN restoration and fixed crop TTA with
  matched controls, job11556720.
- These are unchanged copies of the original metric JSONs, including recorded protocols and
  checkpoint details. `reproducibility/check_results.py` checks AP/final-step/freeze/control
  consistency without datasets or model execution. This is not a new AP computation.
- Optional independent COCO rescoring needs the original Test100 annotation and saved prediction
  files. Those files and full reference audits remain in the working checkouts, not in Git.
- Portable diagnostic path/hash/control-AP arguments retain historical defaults and have local
  argument/root-boundary tests. They were not separately rerun on a GPU.
- Required workflow files are `SKILL.md`, `AGENT.md` and this manifest; `AGENTS.md` is the
  canonical scientific notebook. The original checkpoint remains selected; no bonus is claimed.

## Final reported result

**Selected:** original released HRNet-W48 FiDIP with flip testing. It remains best among all tested
variants; no verified improvement or bonus is claimed. This reproduces checkpoint evaluation,
not the authors' original training process.

- Run ID: baseline GPU `11450265`, CPU checkpoint admission `11450222`; later matched controls
  in `11553889` and `11556720` reproduced the same AP.
- Revision: scored runtime was `7bc14bc` plus reviewed uncommitted fixes (hashes in the baseline
  section/logs); later documentation HEAD `bc66664`. No fabricated clean runtime commit.
  Recorded outputs are copied unchanged under `reproducibility/results/`; the original run inputs
  and source hashes are recorded above. The later source-release commit is `6aa9e7b`; its optional
  diagnostic path/control parameters were locally tested, not a new scored GPU rerun.
- Config: `experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml`, SHA-256
  `cd1151abad97aac80400252c89e000c3ea7fca87975f87b06546e66cc1191157`.
- Dataset: original SyRIP Test100, 100 samples; annotation SHA-256
  `8b299a879a95913483498e3f2115a826ac23ebc5262c3efb8d65f63718e3194c`.
- Checkpoint: `models/hrnet_fidip.pth`, 255,354,649 bytes, SHA-256
  `bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07`.
- Conditions: GT boxes, flip/postprocess on, 384-high × 288-wide, batch16, V100-SXM2-32GB;
  official decoder/OKS rescoring/NMS/COCO evaluator retained. Debug rendering disabled.
- Metrics (percent): AP **93.6071706887983**, AP50/AP75 **98.52475247524751**,
  AR **94.6**, AR50/AR75 **99.0**. Published AP **93.6**, matched after rounding.
- Evidence: `output/baseline-test100-11450265/metrics.json`, saved predictions and
  `log/baseline-evidence-11450265/baseline-11450265.out`. The metric JSON is also available at
  `reproducibility/results/baseline.json`; full logs and predictions remain local.
- Exact evaluation invocation (inside the admitted GPU allocation):

```sh
python -u tools/test_adaptive_model.py \
  --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
  TEST.MODEL_FILE models/hrnet_fidip.pth TEST.USE_GT_BBOX True \
  DATASET.ROOT data/syrip/ OUTPUT_DIR output/baseline-test100-11450265 \
  LOG_DIR log/baseline-test100-11450265 DEBUG.DEBUG False PRINT_FREQ 1
```

`tools/evaluate_baseline.sbatch` also performs independent full-precision rescoring. New reruns
must use their own output/job IDs; see the reproduction guide. The original checkpoint/data remain unchanged.
