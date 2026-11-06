# FiDIP reproduction guide

This project uses the authors' provided code, HRNet-W48/DarkPose configuration and pretrained
FiDIP checkpoint. Original Test100 evaluation reproduced **93.6071706887983 AP**, matching the
published **93.6** after rounding. Five simple follow-ups did not beat it; no bonus is claimed.

The repository contains the ML source, compatibility fixes, scripts, tests, configurations and
recorded metrics. Datasets, weights, saved prediction files and full reference-audit outputs remain
external. The required workflow documents are `SKILL.md`, `AGENT.md` and `MANIFEST.md`;
`AGENTS.md` is the canonical lab notebook.

## 1. Check recorded results

The three files in [results/](results/) are unchanged copies of the original baseline, screen
and diagnostic metric JSONs, including their recorded protocols and checkpoint details.

```sh
python3 reproducibility/check_results.py
```

This requires only Python's standard library and checks the six scores, fixed-step/freeze guards
and matched controls. It does not run a model or recompute AP.

## 2. Acquire immutable inputs separately

Place these **outside Git**, at the indicated paths. No expanded SyRIP data or password is needed.

| Input | Path | SHA-256 |
|---|---|---|
| Original SyRIP | `data/archives/SyRIP.zip` | `1d13e816375988be9bbdab8bf5be6562a5b9bfa83684d57ccbfec9dd619e1692` |
| FiDIP processed training release | `data/archives/syrip_for_train.zip` | `1f267878732fb81776c5c7f153bad45d652cadf5bfd83761971390b64d6cee1b` |
| Original released HRNet | `models/hrnet_fidip.pth` | `bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07` |

Dataset URLs:
- <https://coe.northeastern.edu/Research/AClab/SyRIP/SyRIP.zip>
- <https://coe.northeastern.edu/Research/AClab/SyRIP/syrip_for_train.zip>
- Original checkpoint folder: <https://drive.google.com/drive/folders/108P-1SnTqaj3xNtjYZ1o7T8z6UvUYuiC>

Acquire the original FiDIP HRNet file, not expanded-data weights. The model digest identifies the
user-supplied download; no author-published digest was found. If the official file becomes unavailable,
ask Jack for the checksum-matched original rather than substituting another checkpoint.

For Mac↔Explorer copies use `gladowsky.j@xfer.discovery.neu.edu`, **never** the login alias.
Use `ssh explorer` only for lightweight commands and SLURM. See root `SKILL.md` for exact transfer
examples and policy. Preserve existing work; do not use `rsync --delete` or reset the remote checkout.

## 3. Linux/CUDA environment and data admission

Validated environment: Linux x86-64, Python 3.12.14, torch 2.3.1+cu121, torchvision 0.18.1+cu121,
NumPy 1.26.4, CUDA module 12.1.1; scored runs used V100-SXM2-32GB. `uv.lock` is authoritative.
Do **not** run this project's `uv sync` on macOS.

Explorer uses Jack's personal home (75 GB), not group/project storage. No installation is needed in
his existing validated `.venv`. A new account/check-out must inspect quota and provide uv through an
approved user-local method; then run `uv sync --locked` **inside a bounded CPU allocation**. All
installs/builds/data decoding/evaluation/training stay off login nodes. Adapt SLURM account/partition
eligibility to your account; the recorded default account was `ostadabbas`, QOS `normal`.

Example fresh-checkout setup, submitted from the extracted root (not executed automatically):

```sh
mkdir -p data/archives models output log slurm-logs
sbatch --partition=short --constraint=ib --cpus-per-task=2 --mem=4G --time=00:10:00 \
  --output=slurm-logs/setup-%j.out \
  --wrap='set -euo pipefail; module load cuda/12.1.1; ~/.local/bin/uv sync --locked'
```

After setup COMPLETES successfully, compile/check native extensions and prepare the data:

```sh
build_job=$(sbatch --parsable tools/build_environment.sbatch)
gpu_check_job=$(sbatch --parsable --dependency=afterok:"$build_job" tools/check_environment.sbatch)
data_job=$(sbatch --parsable --partition=short --constraint=ib --cpus-per-task=4 --mem=4G \
  --time=00:05:00 --output=slurm-logs/data-prepare-%j.out \
  --wrap='set -euo pipefail; source .venv/bin/activate; python -B reproducibility/prepare_data.py --extract')
```

`prepare_data.py` verifies both archive hashes, rejects unsafe/duplicate ZIP members, refuses
existing extraction roots and creates the original relative-link layout. Omit `--extract` when
raw trees already exist. Test100 uses the original validation images/labels, not the processed
release's inconsistent `test11.jpg`.

Original validation decoded all 7,657 images and found 46 processed-training dimension mismatches
and four exact train/Test100 duplicates. No labels/images were repaired. Full byte-for-byte
re-audits (`--audit` / `tools/validate_dataset.sbatch`) additionally require the archived auditor
and reference manifests listed in `MANIFEST.md`; those local audit outputs are not shipped in Git.
They are not required by the direct baseline command below. Do not invent a passing audit file.

Native builds use temporary generated C and V100 `sm_70` + PTX. A100 builds can set
`NMS_CUDA_ARCH=80`. Use typed A100/V100 GPU requests, never generic `--gres=gpu:1`.

## 4. Reproduce baseline inference and evaluation

Once native/GPU/data preparation succeeds, run CPU checkpoint admission:

```sh
checkpoint_job=$(sbatch --parsable tools/check_checkpoint.sbatch)
```

It verifies the checkpoint digest, tensor-only strict loading, finite tensors and a CPU forward.
After it COMPLETES successfully, run the authors' entry point **inside a typed A100/V100 GPU
allocation**, not on a login node:

```sh
source .venv/bin/activate
python -u tools/test_adaptive_model.py \
  --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
  TEST.MODEL_FILE models/hrnet_fidip.pth TEST.USE_GT_BBOX True \
  DATASET.ROOT data/syrip/ OUTPUT_DIR output/reproduced-baseline \
  LOG_DIR log/reproduced-baseline DEBUG.DEBUG False PRINT_FREQ 1
```

Use a new output path for each run. This retains the official 384-high × 288-wide input, batch16,
GT boxes, flip testing and original decoder/OKS rescoring/NMS/COCO evaluator. Only output/log
paths, debug rendering and print frequency change.

Expected percentages: **AP93.6071706887983; AP50/AP75 98.52475247524751; AR94.6;
AR50/AR75 99.0**. Medium-area −1 statistics mean no applicable medium examples.

The historical admitted batch workflow is `tools/evaluate_baseline.sbatch`, which also performs
full-precision COCO rescoring. It requires the original full data-admission JSON in addition to
the exported `FIDIP_CHECKPOINT_JOB`; see `MANIFEST.md`. Do not bypass its checks.

With the original Test100 annotation and six saved prediction files at their recorded local paths,
independent CPU rescoring is available inside a CPU allocation with pycocotools:

```sh
python -B reproducibility/check_results.py --rescore
```

Prediction JSONs are not supplied in the lean Git publication; generate them with the evaluation
scripts or obtain the archived outputs separately. Checking recorded metrics is not newly computed AP.

## 5. Optional controlled-experiment reruns

These are the archived experiment commands, **not authorization to launch more runs now**.
CPU preparation additionally needs the original image-hash reference manifest under `data/`;
full preparation/audit outputs are local, not in Git. See `MANIFEST.md` for those prerequisites.
Keep newly generated run directories separate from the original evidence.

```sh
prep_job=$(sbatch --parsable tools/prepare_small_screen.sbatch)
screen_job=$(sbatch --parsable --dependency=afterok:"$prep_job" \
  --export=ALL,FIDIP_SCREEN_PROFILE_JOB="$prep_job" tools/run_small_screen.sbatch)
```

The fresh preparation serializes the approved 1,150-record selection (196 real/954 synthetic), excluding
46 inconsistent samples and four exact Test100 overlaps; source labels and Test100 are not edited.
Both fine-tunes start independently from original weights: 600 steps, batch 8, Adam LR 1e-5,
weight decay 1e-4, seed 7373, balanced replacement sampling, matched batches/augmentations,
FP32/deterministic/TF32 off. The head trial freezes backbone+BN; stage4/head freezes earlier modules.
Final step only is scored; no Test100 early stopping. Flip-off uses the original weights.

Expected AP: head **93.5472078062278**, stage4/head **93.198637527681**, flip-off **91.9158670222938**.
Outputs include raw `final_pose.pth`, resume states at 300/600, protocols, predictions and `results.json`.
Resume states contain optimizer/RNG/protocol and are **not** raw evaluation state dictionaries.

After the NEW screen completes, obtain its checkpoint identity/control AP from its small result JSON
and submit the two inference-only diagnostics:

```sh
read -r trained_sha trained_ap < <(python3 -c \
  'import json,sys; r=json.load(open(sys.argv[1]))["results"]["stage4_head"]; print(r["pose_checkpoint_sha256"], r["AP"])' \
  "output/small-screen-${screen_job}/results.json")
analysis_job=$(sbatch --parsable \
  --export=ALL,FIDIP_DIAG_SCREEN_DIR="output/small-screen-${screen_job}",FIDIP_DIAG_TRAINED_SHA="$trained_sha",FIDIP_DIAG_TRAINED_AP="$trained_ap" \
  tools/analyze_bounded_diagnostics.sbatch)
sbatch --dependency=afterok:"$analysis_job" --export=ALL,FIDIP_DIAG_ANALYSIS_JOB="$analysis_job" \
  tools/run_bounded_diagnostics.sbatch
```

These path/control parameters were added for packaging without changing inference/metric mathematics.
Historical defaults remain valid. The optional path parameters have local tests/static checks,
not a new GPU rerun; original scored-source hashes are recorded in `MANIFEST.md`.

Expected AP: original-BN-buffer restoration **93.35239311605797**; fixed 0.9/1.0/1.1 crop TTA
**93.32060253033816**. Both retain flips and fixed resolution. The restored model keeps all trained
parameters/BN affine weights. The scripts require matched original/trained controls and then stop.
Different platforms may not reproduce exact floating-point results; do not silently relax guards.

## 6. Tests and limitations

```sh
python -B -m unittest discover -s tests -v
```

Tests require the installed scientific dependencies. Mac skips the GNU-timeout-specific fixture.
They do not require document assets, datasets, model weights or a GPU.

Limitations: released-checkpoint evaluation rather than original training; single-seed short trials;
repeated use of only 100 test images; GT boxes; known original train/Test100 overlap, which new
exclusions cannot undo. CNN timing excludes the rest of the application. Keep negative results
without claiming significance, clean generalization or a new best score.

## 7. Checkpoints and durable evidence

The selected original checkpoint is obtainable from the official folder above. Optional trained tensors
are retained at `/home/gladowsky.j/Infant-Pose-Estimation/` on Explorer, not inside Git:

| Artifact | Relative path | SHA-256 |
|---|---|---|
| Head final | `output/small-screen-11553889/head/final_pose.pth` | `9a8f03f65328baceb05ce36a111d9d7b71337d182e4a59dbfa241ee4e5a667ff` |
| Stage4/head final | `output/small-screen-11553889/stage4_head/final_pose.pth` | `b146f8c25d5c45f882603f858d1c532d054e1caedc687fedb41e154fd3be3f6e` |
| BN restored | `output/bounded-diagnostics-11556720/BN_buffers_restored.pth` | `fe0942279648bc817e450823bbf37adad4bae0fe3da7998e6cbd7b2787b664e8` |

Recorded metric/protocol JSONs are included. Saved predictions, full logs and optional trained
weights remain local; obtaining them separately is required for replaying the archived outputs. The screen can regenerate them. If graders request the exact historic tensors,
Jack must arrange a separate authorized course transfer; an account-private path is not a public download.
See `../MANIFEST.md` for input identities, source hashes and run details. Historical SLURM anomalies
are retained, including CPU prep `11553804` script rc0 versus allocation TIMEOUT; it is not relabeled COMPLETED.
