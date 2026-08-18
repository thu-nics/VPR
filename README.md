# Verifiable Process Rewards for Agentic Reasoning

[![Paper](https://img.shields.io/badge/arXiv-2605.10325-b31b1b.svg)](https://arxiv.org/abs/2605.10325)
[![Project Page](https://img.shields.io/badge/Project-Page-0b3d91.svg)](https://thu-nics.github.io/VPR/)
[![Models](https://img.shields.io/badge/HuggingFace-Models-f4c430.svg)](https://huggingface.co/collections/nics-efc/vpr)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

This repository contains the implementation of **Verifiable Process Rewards
(VPR)** on top of [verl-agent](https://github.com/langfengQ/verl-agent) and
[veRL](https://github.com/volcengine/verl). VPR studies structured, long-horizon
agentic reasoning tasks in which a task-grounded symbolic or algorithmic oracle
can score intermediate actions.

The paper evaluates three oracle instantiations:

- **Sokoban:** breadth-first search identifies actions on a shortest solution path.
- **Sudoku:** constraint reasoning distinguishes forced, MRV, legal, and invalid moves.
- **Minesweeper:** posterior mine probabilities identify safe reveals, certain flags,
  and minimum-risk guesses.

## State-group rollout

At every visited state, VPR samples `K=4` candidate responses from the current
policy. Each response is parsed into an environment action and scored by the
oracle. One uniformly sampled maximum-reward candidate is committed to advance
the environment, while every informative, non-padding candidate can train the
policy. Advantages are formed from within-state relative rewards and then
whitened across valid candidates. Equal-reward groups are masked because they
contain no preference signal.

Terminology used throughout the code and documentation:

- **Initial task instance:** one reset environment and its initial prompt.
- **Committed trajectory:** the single environment trajectory produced by VPR's
  committed actions.
- **State group:** the four candidates sampled from one intermediate state.
- **Trajectory group:** the eight complete trajectories sampled by GRPO from one
  initial task instance.

The maximum-reward commit improves exploration of promising successor states but
also changes the visited-state distribution relative to ordinary policy rollout.
`K=4` is the paper's practical exploration/selection-pressure trade-off.

## Installation

The paper environment used Python 3.12, PyTorch 2.8, vLLM 0.11, Ray 2.50,
Transformers 4.57.3, and tensordict 0.10 on CUDA 12. Install platform-compatible
PyTorch and vLLM wheels first, following the upstream verl-agent instructions.

```bash
git clone https://github.com/thu-nics/VPR.git
cd VPR
pip install -e ".[vllm,vpr]"

# Development and CPU tests.
pip install -e ".[test]"
```

For the paper-compatible core package versions, add:

```bash
pip install -c constraints/vpr-training-py312-cu12.txt -e ".[vllm,vpr]"
```

The constraints file records the validated server configuration; it is not a
portable CUDA installer. Compared with upstream verl-agent, the principal VPR
dependency is the pinned `gem-llm` package. Sokoban additionally uses the
upstream-supported `gym-sokoban` environment.

## Quick start

All launchers require an explicit local or Hugging Face model path. Generated
data and run artifacts stay under repository-relative ignored directories.

```bash
export MODEL_PATH=/path/to/Qwen3-4B

python examples/vpr_games/prepare_data.py \
  --env-name vpr_sokoban --train-size 64 --val-size 64 \
  --output-dir examples/vpr_games/data/vpr_sokoban

# Paper VPR configurations: 64 initial tasks, K=4 candidates per visited state.
bash examples/vpr_games/vpr/vpr_sokoban.sh
bash examples/vpr_games/vpr/vpr_sudoku.sh
bash examples/vpr_games/vpr/vpr_minesweeper.sh
```

Useful overrides include `RUN_DIR`, `CUDA_VISIBLE_DEVICES`, `N_GPUS`,
`TP_SIZE`, `TRAIN_STEPS`, `TRAIN_BATCH`, `ROLLOUT_N`, and `RESUME_MODE`.
Set `DRY_RUN=1` to validate a paper-facing launcher without starting training.

## Baselines

```bash
# Outcome-reward GRPO: 32 task instances x 8 complete trajectories.
bash examples/vpr_games/grpo/grpo_sokoban_outcome.sh
bash examples/vpr_games/grpo/grpo_sudoku_outcome.sh
bash examples/vpr_games/grpo/grpo_minesweeper_outcome.sh

# VinePPO.
bash examples/vpr_games/vineppo/vineppo_sokoban.sh
bash examples/vpr_games/vineppo/vineppo_sudoku.sh
bash examples/vpr_games/vineppo/vineppo_minesweeper.sh
```

TicTacToe and Turn-level PPO remain available as additional/legacy
implementations, but they are not part of the paper's main experimental suite.

## Mixed training and evaluation

The mixed comparison holds math exposure fixed at 64 math prompts with eight
responses each. Math+VPR uses 6/8/18 initial Sokoban/Sudoku/Minesweeper prompts
with four candidates at every visited state. Math+GRPO uses 3/4/9 prompts with
eight complete trajectories per prompt. This matches first-decision response
counts, not total sequential rollout compute.

```bash
bash examples/dapo_trainer/run_qwen3_4b_base_math.sh
bash examples/dapo_trainer/run_qwen3_4b_base_vpr_mixed.sh
bash examples/dapo_trainer/run_qwen3_4b_base_games_non_vpr_mixed.sh

cp examples/vpr_games/eval/in_domain_checkpoints.example.env runs/in_domain_checkpoints.env
# Edit the copied paths before sourcing it.
source runs/in_domain_checkpoints.env
MODEL_PATH=/path/to/Qwen3-4B bash examples/vpr_games/eval/eval_in_domain_all.sh
MODEL_MANIFEST=/path/to/models.tsv bash examples/vpr_games/eval/eval_reasoning_all.sh
MODEL_MANIFEST=/path/to/models.tsv bash examples/vpr_games/eval/eval_agentic_ood_all.sh
```

Evaluation launchers preserve protocol manifests, raw generations, source
identity, and resumability under `runs/`. See
[`examples/vpr_games/eval/README.md`](examples/vpr_games/eval/README.md).

## Exploratory tau2-bench extension

The tau2-bench experiment uses a task-grounded expert reference policy with
privileged task specification and reference-resolution guidance. Tool calls use
canonical exact matching and natural-language actions use a semantic matcher.
This is supervised feasibility evidence, not a general no-privilege conversion
from arbitrary policies to process verifiers.

```bash
PYTHON=python bash examples/tau_bench/install_tau2.sh
export OPENROUTER_API_KEY=...
MODEL_PATH=/path/to/Qwen3-8B bash examples/tau_bench/run_tau_vpr.sh
```

Both trained tau2-bench variants are evaluated at step 100. See
[`examples/tau_bench/README.md`](examples/tau_bench/README.md) for the pinned
source revision, qualification protocol, and evaluation procedure.

## Repository map

- `agent_system/environments/env_package/vpr_games/`: game environments and oracles.
- `agent_system/multi_turn_rollout/`: vanilla and state-group rollout.
- `gigpo/core_gigpo.py`: VPR, Turn-level PPO, and VinePPO estimators.
- `verl/trainer/config/vpr_*.yaml`: canonical game configurations.
- `examples/vpr_games/`: data, launchers, smoke checks, and evaluation.
- `examples/dapo_trainer/`: math and mixed-training launchers.
- `examples/tau_bench/`: exploratory tau2-bench protocol.
- `tests/vpr_games/`, `tests/tau_bench/`: regression suites.

## Reproducibility notes

Training the reported 4B experiments used eight GPUs. Full training and vLLM
smoke tests require a CUDA environment; CPU CI covers packaging, parsers,
posterior inference, Tau action validation, evaluation summaries, and command
construction.
See [`docs/reproducibility/provenance.md`](docs/reproducibility/provenance.md)
and [`docs/release_audit.md`](docs/release_audit.md).

## Citation

```bibtex
@misc{yuan2026verifiable,
  title         = {Verifiable Process Rewards for Agentic Reasoning},
  author        = {Huining Yuan and Zelai Xu and Huaijie Wang and Xiangmin Yi and Jiaxuan Gao and Xiao-Ping Zhang and Yu Wang and Chao Yu and Yi Wu},
  year          = {2026},
  eprint        = {2605.10325},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2605.10325}
}
```

## Acknowledgements and license

This work builds on [verl-agent](https://github.com/langfengQ/verl-agent) and
[veRL](https://github.com/volcengine/verl). Their original copyright notices are
preserved. The code is released under the [Apache License 2.0](LICENSE).
