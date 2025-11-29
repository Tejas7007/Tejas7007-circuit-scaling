# Contributing

## Dev workflow
1. Create a branch (feature/..., fix/...)
2. Run `make fmt && make test` before PRs.
3. Prefer small, reviewable PRs with clear experiment IDs.

## Experiments
- Use `results/<exp_id>/` for logs, JSON summaries, and figures.
- Add a short README.md inside each `results/<exp_id>/` with config and findings.
