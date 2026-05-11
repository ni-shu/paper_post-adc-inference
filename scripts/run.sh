#!/usr/bin/env bash
set -eu

TYPE=${1:-all}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

run_exp() {
    uv run python "${ROOT}/scripts/experiment.py" --config "${ROOT}/config/$1"
}

case "${TYPE}" in
    fpr_nsteps)
        run_exp sweep_config_fpr_nsteps.yaml
        ;;
    power)
        run_exp sweep_config_power_diff.yaml
        ;;
    fpr_dim)
        run_exp sweep_config_fpr_dim.yaml
        ;;
    fpr_hyperparameter)
        run_exp sweep_config_fpr_hyperparameter_ucb.yaml
        run_exp sweep_config_fpr_hyperparameter_tpe.yaml
        ;;
    all)
        run_exp sweep_config_fpr_nsteps.yaml
        run_exp sweep_config_power_diff.yaml
        run_exp sweep_config_fpr_dim.yaml
        run_exp sweep_config_fpr_hyperparameter_ucb.yaml
        run_exp sweep_config_fpr_hyperparameter_tpe.yaml
        ;;
    *)
        echo "Unknown TYPE: ${TYPE}"
        echo "Valid types: all fpr_nsteps power fpr_dim fpr_hyperparameter"
        exit 1
        ;;
esac
