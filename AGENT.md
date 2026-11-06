# FiDIP course submission — agent workflow

This is the course-required entry point, not a second lab notebook or a symlink.
The canonical context and chronological record remain in [AGENTS.md](AGENTS.md).

## Start here

- [Reproduction guide](reproducibility/README.md): inputs, setup, commands and expected results.
- [Recorded results](reproducibility/results/): original baseline and experiment metric JSONs.
- [SKILL.md](SKILL.md): Explorer operations and safety rules.
- [ENVIRONMENT.md](ENVIRONMENT.md): locked Linux/CUDA environment and validation.
- [MANIFEST.md](MANIFEST.md): data/model hashes, run IDs, results and artifact locations.
- [Report](FiDIP_Report.pdf): concise methods, quantitative and visual analysis, limitations and AI-tool reflection.

## Outcome and scope

The released HRNet-W48 checkpoint reproduced **93.6071706887983 AP** on original SyRIP Test100,
matching the published **93.6** after rounding. Two fine-tunes and three inference/normalization
variants did not improve it. The original checkpoint with flip testing remains the final selection.
I used the authors' provided code, configuration and pretrained HRNet-W48/DarkPose model;
I did not develop a new architecture or compare backbones. My work was running and checking the
existing pipeline, then trying a few simple changes. This reproduces released-checkpoint evaluation,
not the original training process. No bonus is claimed.

## Agent use and human decisions

I used an LLM coding assistant in the Pi harness for much of the low-level coding and operations:
getting the environment working, checking assets/checkpoints, writing cluster scripts and tests,
running approved experiments and collecting evidence.

I set the direction, reviewed the proposed work and results, and approved the next steps. I also
stopped or redirected work, including narrowing the experiment plan, limiting compute use,
and deciding not to chase the bonus. The agent did not have open-ended permission to
keep launching runs. Code checks also caught a balancing proposal that the authors' sampler
already implemented. The final review and submission are my responsibility.

No further training, cluster changes, Git commit/push or submission is authorized by these instructions.
