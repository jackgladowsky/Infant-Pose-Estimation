# FiDIP Reproduction Manifest

This file inventories every input and output needed to reproduce reported results. Update paths,
checksums, versions, and run IDs whenever an artifact is obtained or produced.

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
- Home was the earlier interim location; after the 2026-10-22 policy review, further installs,
  bulk downloads, and moves are paused pending Jack's agreement on approved storage. Prefer
  `/projects` for large environments/caches and research artifacts; ask instructor/RC if unavailable.
  Earlier notes recorded a 75 GB home quota; verify current usage/limits from a compute allocation.
  Expected artifact paths below resolve relative to the existing checkout; no datasets or weights
  acquired yet. No environment migration or deletion has been performed.
- Setup/docs and the retrieved lockfile were committed and pushed to Jack's fork at `8f4c497`,
  then fast-forwarded on Explorer on 2026-10-22. The remote checkout was clean after synchronization.
- Original remote uncommitted setup is preserved in Git stash entry
  `FiDIP setup preserved before fork sync 8f4c497` (`stash@{0}` when created); do not drop it automatically.
- Effective read/write/traverse access to `/projects/aclab` is verified. Confirm instructor approval
  and agree on the per-user path before creating directories or moving/installing anything.
- `/courses/EECE7398.202710` exists but this account has no read/write/traverse access.
- Latest read-only inspection found a dependency installation log and `uv.lock`; GPU validation
  and baseline evaluation remain unverified. No queued jobs were observed.

## uv setup (installed-package log present; validation pending)

- `pyproject.toml`: Linux x86-64 training/evaluation dependencies, explicit CUDA 12.1 Torch index.
- `.python-version`: Python 3.12; exact installed interpreter must be verified.
- `.venv/`: present remotely; not yet verified.
- `uv.lock`: retrieved locally from Explorer on 2026-10-22 (41,582 bytes), with matching SHA-256
  `a9d18d4df5ab965229e9205e488b8b9938a773f3326db7c4bf281ad29dbfac4a`.
  Local/remote `pyproject.toml` SHA-256 also matches:
  `44900cbb39e036cb463ee46591230a4fa83cc3a4e7846cbb5032817f0a783dbb`.
- `ENVIRONMENT.md`: setup and validation instructions.
- `tools/check_environment.py` and `tools/check_environment.sbatch`: pending GPU validation.
- `log/uv-sync.log`: installation log on Explorer; lists installed PyTorch 2.3.1+cu121 and other
  dependencies. No active sync process observed during the 2026-10-22 review.

## Project context and course artifacts

- `AGENTS.md`: canonical agent context and chronological research notebook.
- `AGENT.md`: relative symlink to `AGENTS.md`, retaining the course-required artifact name.
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

Datasets must not be committed. Prefer persistent Explorer project storage and expose the expected
repository path through a symlink if needed.

| Dataset | Expected repository path | Actual storage path | Access/status | Checksum/version |
|---|---|---|---|---|
| SyRIP annotations | `data/syrip/annotations/` | TODO | Missing; restricted download | TODO |
| SyRIP infant train images | `data/syrip/images/train_infant/` | TODO | Missing | TODO |
| SyRIP pre-infant train images | `data/syrip/images/train_pre_infant/` | TODO | Missing; may require author contact | TODO |
| SyRIP validation images | `data/syrip/images/validate_infant/` | TODO | Missing | TODO |

Expected annotation files:

- `person_keypoints_train_pre_infant.json`
- `person_keypoints_train_infant.json`
- `person_keypoints_validate_infant.json`

## Downloaded model inputs

Model files must not be committed.

| Model | Expected repository path | Source/status | SHA-256 |
|---|---|---|---|
| Pretrained FiDIP HRNet | `models/hrnet_fidip.pth` | Missing; FiDIP Google Drive folder linked from `README.md` | TODO |
| Pretrained FiDIP MobileNet | `models/mobile_fidip.pth` | Missing; FiDIP Google Drive folder linked from `README.md` | TODO |
| DarkPose HRNet-W48 | `models/coco/w48_384x288.pth` | Missing; TGA Google Drive folder linked from `README.md` | TODO |

## Reference baseline

The repository reports the following SyRIP Test100 HRNet-W48/DarkPose metrics:

| Method | AP | AP50 | AP75 | AR | AR50 | AR75 |
|---|---:|---:|---:|---:|---:|---:|
| DarkPose + fine-tune | 92.7 | 98.5 | 98.5 | 93.9 | 99.0 | 99.0 |
| DarkPose + FiDIP | **93.6** | 98.5 | 98.5 | 94.6 | 99.0 | 99.0 |

These are published reference values, not yet locally reproduced.

Baseline evaluation command:

```bash
python tools/test_adaptive_model.py \
  --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
  TEST.MODEL_FILE models/hrnet_fidip.pth TEST.USE_GT_BBOX True
```

## Produced experiments and checkpoints

| Run | Git commit | Config | Dataset version | Job/W&B ID | Metrics | Checkpoint/log path | Status |
|---|---|---|---|---|---|---|---|
| Baseline reproduction | TODO | HRNet config above | TODO | TODO | TODO | TODO | Not started |

## Final reported result

Not selected. Fill this section only after results are reproduced and verified:

- Run ID: TODO
- Git commit: TODO
- Config: TODO
- Dataset version: TODO
- Checkpoint path and SHA-256: TODO
- AP/AP50/AP75/AR: TODO
- Evaluation command and log path: TODO
