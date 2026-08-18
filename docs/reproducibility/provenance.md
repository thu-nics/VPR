# Release provenance

This public release was prepared in an isolated repository. It does not modify
the development checkout or any experiment artifacts.

- Upstream verl-agent baseline: `234a913609ef820de826f2276b7b912e4b8b8410`
- VPR development snapshot: `48cb073905d2db4c93651d42db74342c6e0b73dc`
- Release branch: `release/vpr-neurips-2026`
- Python used by the training-server snapshot: 3.12
- Snapshot export date: 2026-08-18

The development commits through the snapshot above are represented by one
squashed implementation commit. The release-cleanup commit adds public
documentation, portable paths, packaging, canonical configs, CI, and audit
material.

Two uncommitted development files were intentionally considered:

- `examples/vpr_games/vpr/vpr_minesweeper.sh`: included after syntax and
  configuration validation; it adds explicit context, dynamic-batch, sequence
  parallel, and action-format controls.
- `examples/vpr_games/eval/eval_reasoning_all.sh`: included after portable-path
  cleanup and dry-run validation.

The unfinished `direct_agentic_ood.py`, internal agent instructions, future
research plans, paper PDFs, internal audits, `runs/`, datasets, logs, model
weights, and checkpoints are excluded.

The original 314-package server `pip freeze` was used as evidence but is not an
install file: it includes unrelated editable OpenEnv/AWM packages and an
editable self-install of an older upstream verl-agent checkout. The curated
constraints file records the core training compatibility versions without
those unrelated packages.
