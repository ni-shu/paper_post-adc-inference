#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MAX_JOBS", "1")

import argparse
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from si_state import (
    set_candidates,
    set_randomized,
    set_seed,
    RANDOMIZED_METHODS,
    SICoreInverseCDFNaNError,
)
from si_core import (
    SIIntervalViolationError,
    IntervalMeanWindowNotFoundError,
    InverseCDFNaNError,
)
from bo_models import GPRegression, RBF, TPEModel, ModelBase, acquire, select_window
from inference import MeanDiff, ablated_stat, infer, summarize
from config import expand_configs, identify_keys, f_star, grid


def run_pipeline(cfg: dict, seed: int, methods: list[str]) -> dict[str, dict[str, dict]]:
    set_seed(seed)
    randomized = bool(set(methods) & RANDOMIZED_METHODS)
    set_randomized(randomized, cfg["randomized_var"] if randomized else None)
    if randomized and set(methods) - RANDOMIZED_METHODS:
        raise AssertionError("randomized and non-randomized methods must be run in separate pipelines")

    rng = np.random.default_rng(seed)
    d = int(cfg["x_dim"])
    Xc = grid(d, int(cfg["points_per_axis"]))
    truth = f_star(Xc, cfg["f_star_spec"])
    obs = truth + rng.normal(0.0, math.sqrt(float(cfg["obs_noise_variance"])), size=truth.size)
    set_candidates(Xc)

    initial_n, steps = int(cfg["initial_n"]), int(cfg["num_steps"])
    if initial_n + steps > Xc.shape[0]:
        raise ValueError("initial_n + num_steps exceeds candidate count")
    model: ModelBase = TPEModel(cfg["obs_noise_variance"]) if cfg["acquisition"] == "tpe" else GPRegression(RBF(cfg["rbf_length_scale"], cfg["rbf_variance"]), cfg["obs_noise_variance"])
    idx0 = rng.choice(Xc.shape[0], size=initial_n, replace=False)
    model.set_XY(Xc[idx0], obs[idx0])
    chosen = []
    for _ in range(steps):
        j = acquire(model, cfg)
        model.add_XY(Xc[j], obs[j])
        chosen.append(j)

    assert model.X is not None and model.y is not None
    L = tuple(float(v) for v in cfg["selection_spec"]["window_length"])
    I = select_window(model.y, model.X, L, largest=True)
    J = select_window(model.y, model.X, L, largest=False, exclude=I)
    stat = MeanDiff(model.y, I, J)
    obs_indices = np.concatenate([idx0, np.asarray(chosen, dtype=int)])
    log_m = float(steps) * math.log(float(Xc.shape[0])) + float(steps + initial_n) * math.log(3.0)
    alt, cl = str(cfg["alternative"]).lower(), float(cfg["confidence_level"])

    out = {}
    for method in methods:
        stat_i = ablated_stat(stat, method) if method in {"randomized_path_only", "randomized_selector_only"} else stat
        actual = "randomized" if method in RANDOMIZED_METHODS else method
        res = infer(stat_i, actual, float(cfg["obs_noise_variance"]), alt, log_m if method == "bonferroni" else None)
        out[method] = {"diff": summarize(res, cl)}
        out[method]["diff"]["theta_true"] = stat.true_value(truth[obs_indices])
    return out


def run_one_with_retries(cfg: dict, run_id: int, seed: int, methods: list[str]):
    attempts, seed_i = 0, int(seed)
    retry_errors = (SIIntervalViolationError, IntervalMeanWindowNotFoundError, InverseCDFNaNError, SICoreInverseCDFNaNError)
    while True:
        try:
            return run_pipeline(cfg, seed_i, methods), seed_i
        except retry_errors:
            if bool(cfg["enforce_complete"]) and attempts < int(cfg["max_retries_per_run"]):
                attempts += 1
                seed_i += 1
                continue
            return None, seed_i


def run_experiment(cfg: dict) -> dict[str, str]:
    methods = [str(m) for m in cfg["methods"]]
    artifact_path = str(cfg["artifact_path"])
    keys = identify_keys(cfg, artifact_path)
    base_seed = int(cfg["seed"])

    def one(i_seed):
        run_id, seed = i_seed
        merged, available = {}, set()
        for group in ([m for m in methods if m not in RANDOMIZED_METHODS], [m for m in methods if m in RANDOMIZED_METHODS]):
            if not group:
                continue
            res, _ = run_one_with_retries(cfg, run_id, seed, group)
            if res is not None:
                merged.update(res); available.update(group)
        rows = []
        for method in methods:
            if method not in available:
                rows.append(None); continue
            s = merged[method]["diff"]
            row = {"method": method, "p_value": np.float64(s["p_value"]), "ci_low": np.float64(s["ci90"][0]), "ci_high": np.float64(s["ci90"][1]), "theta_true": np.float64(s["theta_true"])}
            row.update(keys)
            rows.append(row)
        return tuple(rows)

    dataset = [(i, base_seed + i) for i in range(int(cfg["n_runs"]))]
    streams = Parallel(n_jobs=int(cfg["num_worker"]))(delayed(one)(d) for d in tqdm(dataset, total=len(dataset), file=sys.stdout))
    buffers = {m: [] for m in methods}
    for method, stream in zip(methods, zip(*streams)):
        buffers[method].extend(r for r in stream if r is not None)
    missing = [f"Experiment incomplete for method '{m}': expected {cfg['n_runs']} results, got {len(v)}." for m, v in buffers.items() if len(v) != int(cfg["n_runs"])]
    if missing:
        raise RuntimeError("\n".join(missing))
    rows = [r for v in buffers.values() for r in v]
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cols = ["method", "p_value", "ci_low", "ci_high", "theta_true"] + [k for k in keys if k in df.columns]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols]
    for col in ["p_value", "ci_low", "ci_high", "theta_true"]:
        df[col] = df[col].astype(np.float64)
    df.to_parquet(artifact_path, engine="pyarrow", index=False)
    return {"all": artifact_path}


def run_sweep(config_path: Path, dry_run: bool = False, sleep: float = 0.0, keep_going: bool = False, outdir: Optional[str] = None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning,ignore::UserWarning")
    cfgs = expand_configs(config_path, outdir=outdir)
    print(f"Prepared {len(cfgs)} run(s) from {config_path}")
    for i, cfg in enumerate(cfgs, 1):
        print(f"[{i}/{len(cfgs)}] artifact_path={cfg['artifact_path']}")
    if dry_run:
        return []
    results, failures = [], []
    for i, cfg in enumerate(cfgs, 1):
        print("\n" + "#" * 60)
        print(f"Running: {i}/{len(cfgs)}")
        print(f"artifact_path: {cfg['artifact_path']}")
        try:
            paths = run_experiment(cfg)
            print(f"Saved: {paths}")
            results.append(paths)
        except Exception as e:
            failures.append((str(cfg["artifact_path"]), e))
            print(f"ERROR: {cfg['artifact_path']}: {type(e).__name__}: {e}", file=sys.stderr)
            if not keep_going:
                raise
        if sleep > 0:
            time.sleep(float(sleep))
    if failures and keep_going:
        raise RuntimeError("\n".join(f"- {p}: {type(e).__name__}: {e}" for p, e in failures))
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config_pos", nargs="?")
    p.add_argument("--config", dest="config_opt")
    p.add_argument("--outdir")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--keep-going", action="store_true")
    a = p.parse_args()
    config = a.config_opt or a.config_pos
    if not config:
        p.error("provide a YAML path either positionally or with --config")
    run_sweep(Path(config), dry_run=a.dry_run, sleep=a.sleep, keep_going=a.keep_going, outdir=a.outdir)


if __name__ == "__main__":
    main()
