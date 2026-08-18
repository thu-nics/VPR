# Contributing

Please open an issue before a large behavioral change. Keep patches focused and
preserve upstream verl behavior outside the VPR path.

For VPR environment changes, trace the complete path from prompt and parser
through the worker, manager, rollout metadata, reward manager, advantage
estimator, loss mask, metrics, and evidence. State-group changes must preserve
same-state candidates, one committed successor, explicit group IDs, padding
exclusion, and equal-reward masking.

Before submitting a change, run the smallest relevant tests, then the affected
VPR or Tau suite. Run `ruff check` and `ruff format --check` only on changed
Python files. GPU-only checks may be reported as unavailable, but should not be
replaced with mocks that bypass the behavior under test.
