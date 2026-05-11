#!/usr/bin/env bash
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

uv run python "${ROOT}/scripts/visualize.py" fpr_nsteps
uv run python "${ROOT}/scripts/visualize.py" fpr_dim
uv run python "${ROOT}/scripts/visualize.py" fpr_hyperparameter
uv run python "${ROOT}/scripts/visualize.py" power
