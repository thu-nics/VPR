# VPR game environments

This directory contains the paper-facing Sokoban, Sudoku, and Minesweeper
training and evaluation entry points. Each environment uses a Markovian prompt:
the policy observes the current state rather than a concatenated interaction
history (`env.history_length=0`). Actions use the final `<action>...</action>`
block unless a launcher explicitly selects the boxed format.

## Environments and oracles

| Environment | Paper evaluation setting | Oracle process signal |
|---|---|---|
| Sokoban | 7x7, 3 boxes, 36 steps | BFS actions on a shortest solution path |
| Sudoku | 9x9, 40 blanks, 100 steps | forced/MRV/legal/wrong/invalid tiers |
| Minesweeper | 5x5, 5 mines, 25 steps | posterior-safe reveals, certain flags, and minimum-risk guesses |

`reward_mode` controls what the environment emits. `algorithm.adv_estimator`
independently controls how those rewards become advantages. The main VPR path
uses dense `oracle` rewards and `adv_estimator=vpr`; the GRPO baseline uses
sparse `outcome` rewards and `adv_estimator=grpo`.

## Rollout units

- An **initial task instance** is one reset environment and initial prompt.
- VPR produces one **committed trajectory** per task instance.
- At each visited state, VPR samples a **state group** of four candidates.
- GRPO samples a **trajectory group** of eight complete trajectories from one
  initial task instance.

All VPR candidates share the same source state and receive their own parsed
action, oracle reward, and metadata. One maximum-reward candidate is uniformly
selected among ties to advance the environment. Every informative non-padding
candidate can enter the policy loss. Equal-reward state groups are masked.

## Installation and data

```bash
pip install -e ".[vllm,vpr]"
export MODEL_PATH=/path/to/Qwen3-4B

python examples/vpr_games/prepare_data.py \
  --env-name vpr_sokoban --train-size 64 --val-size 64 \
  --output-dir examples/vpr_games/data/vpr_sokoban
```

The parquet rows are trigger instances. Environment managers construct the
actual current-state prompt during rollout.

## Training

```bash
# VPR: 64 initial tasks; K=4 candidates per visited state.
bash examples/vpr_games/vpr/vpr_sokoban.sh
bash examples/vpr_games/vpr/vpr_sudoku.sh
bash examples/vpr_games/vpr/vpr_minesweeper.sh

# Outcome GRPO: 32 initial tasks; G=8 full trajectories per task.
bash examples/vpr_games/grpo/grpo_sokoban_outcome.sh
bash examples/vpr_games/grpo/grpo_sudoku_outcome.sh
bash examples/vpr_games/grpo/grpo_minesweeper_outcome.sh

# VinePPO baselines.
bash examples/vpr_games/vineppo/vineppo_sokoban.sh
bash examples/vpr_games/vineppo/vineppo_sudoku.sh
bash examples/vpr_games/vineppo/vineppo_minesweeper.sh
```

Launchers require `MODEL_PATH`. They default to `PYTHON=python` and write into
the repository-relative `runs/` directory. Common overrides include
`CUDA_VISIBLE_DEVICES`, `N_GPUS`, `TP_SIZE`, `RUN_DIR`, `TRAIN_STEPS`,
`TRAIN_BATCH`, `ROLLOUT_N`, and resume controls. `DRY_RUN=1` validates the
paper-facing configuration without starting a model.

## Evaluation

```bash
MODEL_MANIFEST=/path/to/models.tsv bash examples/vpr_games/eval/eval_in_domain_all.sh
MODEL_MANIFEST=/path/to/models.tsv bash examples/vpr_games/eval/eval_reasoning_all.sh
MODEL_MANIFEST=/path/to/models.tsv bash examples/vpr_games/eval/eval_agentic_ood_all.sh
```

The evaluation pipeline records protocol hashes, source identity, raw outputs,
completion markers, and summary files. Reusing a `RUN_DIR` resumes only when the
stored protocol matches. See [`eval/README.md`](eval/README.md).

## Tests

```bash
python -m pytest tests/vpr_games/test_parser.py -q
python -m pytest tests/vpr_games/test_vpr_advantage.py -q
python -m pytest tests/vpr_games/test_mixed_vpr.py -q
python -m pytest tests/vpr_games/ -q
```

GPU smoke launchers live under `smoke/`; they require the full CUDA/vLLM stack
and validate emitted evidence in addition to the training exit code.

## Additional implementations

TicTacToe, Turn-level PPO, and their smoke scripts are retained for research
continuity. They are not part of the paper's main three-environment results or
Quick Start protocol. TicTacToe's optional MCTS opponent requires
`pip install -e ".[legacy-vpr]"`.
