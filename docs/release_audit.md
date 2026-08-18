# Public release audit

Audit date: 2026-08-18. The release was prepared on macOS in an isolated
checkout; no source development checkout, model, dataset, log, or checkpoint
was modified. No dedicated local training environment was installed, and no
veRL training job was attempted on this Mac.

## Included scope

- VPR environments, state-group rollout, advantage estimation, and evidence.
- GRPO and VinePPO paper baselines.
- Mixed math/game training and OOD evaluation.
- Exploratory Tau Bench training and evaluation.
- Additional TicTacToe and Turn-level PPO implementations, clearly marked as
  outside the main paper suite.

## Excluded artifacts

- Model weights and checkpoints.
- Generated datasets, raw generations, logs, caches, and smoke evidence.
- Internal paper PDFs, rebuttal material, audits, and future research plans.
- The unfinished direct agentic evaluator prototype.

## Packaging and dependencies

- A Python 3.12 wheel built successfully with 451 archive entries.
- The wheel contains `verl`, `agent_system`, `gigpo`, all three game configs,
  the mixed config, and the Tau configs.
- A no-dependency temporary-target check verified that `agent_system` and
  `gigpo` import and that the installed `verl` module is discoverable; the
  temporary target was removed after inspection. A dependency-complete
  `import verl` was not used as a release gate on macOS because that import
  eagerly requires the Torch/Ray/TensorDict/Transformers training stack. The
  CPU CI verifies package discovery; the CUDA-server check below remains
  required before publishing.
- `setup.py`, `pyproject.toml`, and `requirements.txt` now agree on the public
  packages and supported version ranges. `constraints/vpr-training-py312-cu12.txt`
  records the curated server snapshot rather than publishing the unrelated
  314-package freeze.

## Static and CPU checks

- `git diff --check`: passed for both release repositories.
- Python `compileall`: passed for VPR games, Tau, mixed/evaluation tools, and
  `gigpo` under Python 3.12.
- `bash -n`: passed for every shell script under the VPR game, mixed DAPO, and
  Tau example directories.
- Ruff fatal-error lint (`E9,F63,F7,F82`): passed for the public CPU CI scope.
  Ruff's full-fork formatting profile was intentionally not applied to large
  inherited verl-agent modules.
- CPU regression tests: 40 passed (`parser`, Minesweeper posterior,
  in-domain summary, and Tau action parsing/validation).
- The remaining Tau protocol/evaluator tests require the pinned Tau checkout
  and Ray runtime; the full VPR advantage/padding suites require Torch,
  TensorDict, Ray, and GEM. They are listed as server checks below rather than
  replaced with broad mocks.

## Launcher and protocol checks

- `DRY_RUN=1` passed for VPR, GRPO, and VinePPO on Sokoban, Sudoku, and
  Minesweeper.
- `DRY_RUN=1` passed for math-only DAPO, mixed VPR, and mixed GRPO. The emitted
  budgets were VPR games `6/8/18 x 4`, GRPO games `3/4/9 x 8`, and math
  `64 x 8`, all for 100 updates.
- `DRY_RUN=1` passed for in-domain, general-reasoning, ALFWorld/WebShop, and
  native Tau evaluation using temporary manifests. Tau model examples use
  checkpoint step 100.
- Portable SHA-256 handling and dry-run lock bypass were checked with the
  system Bash 3.2 as well as `bash -n`; production runs still retain exclusive
  `flock` locking on Linux.

## Path, secret, and artifact scan

- No developer-specific cluster, virtual-environment, or macOS home path
  remains in the VPR, mixed, Tau, related tests, public configs, README, or
  release docs.
- No private-key marker was found in the public scope.
- No model weight, checkpoint, run log, evaluation output, or local protocol
  manifest is included.
- The only tracked file above 10 MB is the pre-existing WebShop Chromium
  driver required by that upstream environment; it is not a generated model
  or experiment artifact.

## Project page checks

- The page was parsed as HTML and inspected in a real Chromium browser at a
  desktop viewport and at 390 x 844 mobile resolution.
- Navigation, responsive one-column layouts, horizontally scrollable tables,
  all four figures, and the Paper/Code/Models resource cards rendered without
  console errors.
- The stale PNG beside `vpr.pdf` was detected during visual QA and replaced by
  a fresh render of the current Sokoban/Sudoku/Minesweeper PDF.
- Page text and assets contain no TicTacToe, MCTS, MC-PR, Turn-Level
  Optimization, or evaluation-curve reference.

## Required CUDA-server checks before publishing

- Install the ordinary upstream training stack plus `.[vpr]`, then verify
  dependency-complete imports and Hydra composition on Python 3.12/CUDA 12.
- Run the full VPR game and Tau test suites listed in `AGENTS.md`.
- Run one state-group smoke update for each game and validate emitted evidence.
- Run at least one mixed and one Tau smoke job with the pinned external data
  and Tau checkout. A successful process exit must not replace evidence and
  protocol verification.

GPU smoke training was not run on this Mac and is not represented as passed.
