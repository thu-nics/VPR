# Evaluation

All evaluators write protocol metadata, model identity, raw records, completion
markers, and summaries under an ignored `runs/` directory. Reusing `RUN_DIR`
resumes only when the stored protocol remains compatible.

## In-domain games

`eval_in_domain_all.sh` evaluates the common base model plus any explicitly
registered Sokoban, Sudoku, and Minesweeper checkpoints. Historical local run
paths are deliberately not embedded in the public script.

```bash
cp examples/vpr_games/eval/in_domain_checkpoints.example.env \
   runs/in_domain_checkpoints.env
# Edit the copied paths, then:
source runs/in_domain_checkpoints.env
MODEL_PATH=/path/to/Qwen3-4B \
RUN_DIR=runs/eval_in_domain \
  bash examples/vpr_games/eval/eval_in_domain_all.sh
```

Unset checkpoint variables are skipped; the base model is always included.
The paper protocol uses five environment seeds and 100 games per seed.

## General reasoning

`eval_reasoning_all.sh` accepts a tab-separated model manifest with columns
`model_id`, `source_type`, `source_path`, and `benchmarks`. `source_type` is
`hf` or `verl_fsdp`; benchmark names are comma-separated.

```bash
MODEL_MANIFEST=/path/to/reasoning_models.tsv \
EVALSCOPE_ROOT=/path/to/eval-scope \
RUN_DIR=runs/eval_reasoning \
  bash examples/vpr_games/eval/eval_reasoning_all.sh
```

The default paper protocol uses 16K response tokens, a 20K model context,
temperature 0.6, top-p 0.95, top-k 20, and Qwen3 thinking mode. FSDP models are
merged once into an ignored cache. Set `DRY_RUN=1` to validate model and
benchmark rows and emit per-benchmark commands without launching vLLM.

## ALFWorld and WebShop

`eval_agentic_ood_all.sh` evaluates a TSV model manifest on ALFWorld
`valid_unseen` and the WebShop test split. The paper result uses the boxed
action format, 16K prompt/8K response budgets, middle truncation, five ALFWorld
seeds, and three WebShop seeds.

```bash
cp examples/vpr_games/eval/agentic_ood_models.example.tsv runs/agentic_models.tsv
ALFWORLD_DATA=/path/to/alfworld/data \
WEBSHOP_DATA_DIR=/path/to/webshop/data \
MODEL_MANIFEST=runs/agentic_models.tsv \
ACTION_FORMATS=boxed \
RUN_DIR=runs/eval_agentic_ood \
  bash examples/vpr_games/eval/eval_agentic_ood_all.sh
```

Relative model paths resolve from the repository root. The evaluator never
stops unrelated GPU processes; it waits for the explicitly selected devices.
Set `DRY_RUN=1` to validate and print jobs without launching inference.
