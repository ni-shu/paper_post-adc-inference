from __future__ import annotations

import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from si_state import ALL_DEFAULT_METHODS


def points_per_axis(d: int) -> int:
    return 1024 if d == 1 else 32 if d == 2 else 10


def rbf_length(d: int) -> float:
    return {1: 0.1, 2: 0.1414, 3: 0.1732}[int(d)]


def window_length(d: int) -> tuple[float, ...]:
    return {1: (0.2,), 2: (0.4472, 0.4472), 3: (0.5848, 0.5848, 0.5848)}[int(d)]


def grid(d: int, n_axis: int) -> np.ndarray:
    axes = [np.linspace(0.0, 1.0, int(n_axis)) for _ in range(int(d))]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], axis=1)


def scaled_pm1(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    lo, hi = float(np.min(y)), float(np.max(y))
    return np.zeros_like(y) if hi == lo else (y - 0.5 * (hi + lo)) / (0.5 * (hi - lo))


def f_star(X: np.ndarray, spec: dict) -> np.ndarray:
    t = str(spec.get("type", "sine_mix")).lower()
    a = float(spec.get("a", 1.0))
    if t in {"sine_mix", "sinemix"}:
        y = np.mean(float(spec.get("a", 0.0)) * np.sin(20.0 * X) + float(spec.get("b", 0.0)) * np.sin(np.pi * X), axis=1)
        if bool(spec.get("normalize", False)):
            y = (y - np.mean(y)) / (np.std(y) or 1.0)
        return y.astype(float)
    if t in {"negative_forrester", "forrester", "forrester1d"}:
        y = np.mean((6.0 * X - 2.0) ** 2 * np.sin(12.0 * X - 4.0), axis=1)
        if t == "negative_forrester":
            y = -scaled_pm1(y)
        return (a * y).astype(float)
    funcs = {
        "sinc": np.sinc(10.0 * (X - 0.5)),
        "cosine": -np.cos(2.0 * np.pi * X),
        "chirp": np.sin(2.0 * np.pi * X**2),
        "gaussian_bump": np.exp(-(X - 0.7) ** 2 / (2.0 * 0.08**2)),
        "nonsmooth_peak": 1.0 - np.abs(X - 0.4),
    }
    aliases = {"g1": "sinc", "g2": "cosine", "g3": "chirp", "g4": "gaussian_bump", "g5": "nonsmooth_peak"}
    key = aliases.get(t, t)
    if key not in funcs:
        if key in {"constant", "const"}:
            return np.full(X.shape[0], a * float(spec.get("value", 0.0)))
        raise ValueError(f"unknown f_star_spec.type={t!r}")
    return (a * scaled_pm1(np.mean(funcs[key], axis=1))).astype(float)


def identify_keys(cfg: dict, artifact_path: str) -> dict:
    stem = Path(artifact_path).stem
    out = {"acquisition": str(cfg.get("acquisition", "ucb"))}
    if stem.startswith("synthetic_fpr_dim__"):
        out["x_dim"] = int(cfg["x_dim"])
    elif stem.startswith("synthetic_fpr_nsteps__"):
        out["num_steps"] = int(cfg["num_steps"]); out["x_dim"] = int(cfg["x_dim"])
    elif stem.startswith("synthetic_hyper_ucb__"):
        out["kappa"] = float(cfg["kappa"]); out["x_dim"] = int(cfg["x_dim"])
    elif stem.startswith("synthetic_hyper_tpe__"):
        out["tpe_gamma"] = float(cfg["tpe_gamma"]); out["x_dim"] = int(cfg["x_dim"])
    elif stem.startswith("synthetic_power__"):
        fs = cfg["f_star_spec"]
        out["f_star_spec.type"] = str(fs["type"]); out["f_star_spec.a"] = float(fs["a"]); out["x_dim"] = int(cfg["x_dim"])
    else:
        raise ValueError(f"unknown artifact filename: {artifact_path}")
    return out


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def undot(flat: dict) -> dict:
    out = {}
    for k, v in flat.items():
        cur = out
        parts = str(k).split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def expand_grid(g: dict):
    if not g:
        yield {}; return
    keys = list(g.keys())
    for vals in itertools.product(*(g[k] for k in keys)):
        yield dict(zip(keys, vals))


def family(config_path: Path) -> str:
    s = config_path.stem
    if "hyperparameter_ucb" in s or ("hyper" in s and "ucb" in s): return "hyper_ucb"
    if "hyperparameter_tpe" in s or ("hyper" in s and "tpe" in s): return "hyper_tpe"
    if "nsteps" in s: return "fpr_nsteps"
    if "power" in s: return "power"
    if "dim" in s or "dimension" in s: return "fpr_dim"
    raise ValueError(f"cannot infer artifact family from {config_path}")


def fmt_float(x: float, ndigits: int) -> str:
    return f"{float(x):.{ndigits}f}".rstrip("0").rstrip(".")


def artifact_name(cfg: dict, fam: str) -> str:
    acq = cfg.get("acquisition", "ucb")
    if fam == "fpr_dim":
        return f"runs/synthetic_fpr_dim__acq-{acq}__xdim-{int(cfg['x_dim'])}.parquet"
    if fam == "fpr_nsteps":
        return f"runs/synthetic_fpr_nsteps__acq-{acq}__nsteps-{int(cfg['num_steps'])}.parquet"
    if fam == "hyper_ucb":
        return f"runs/synthetic_hyper_ucb__kappa-{fmt_float(cfg['kappa'], 1)}.parquet"
    if fam == "hyper_tpe":
        return f"runs/synthetic_hyper_tpe__gamma-{fmt_float(cfg['tpe_gamma'], 2)}.parquet"
    if fam == "power":
        fs = cfg["f_star_spec"]
        return f"runs/synthetic_power__acq-{acq}__signal-{fs['type']}__a-{fmt_float(fs['a'], 1)}.parquet"
    raise ValueError(f"unknown artifact family: {fam}")


def defaults(cfg: dict) -> dict:
    c = dict(cfg)
    c.setdefault("n_runs", 1000); c.setdefault("num_worker", 1); c.setdefault("methods", ALL_DEFAULT_METHODS)
    c.setdefault("statistic", "diff"); c.setdefault("seed", 0); c.setdefault("num_steps", 50)
    c.setdefault("acquisition", "ucb"); c.setdefault("kappa", 2.0)
    c.setdefault("tpe_gamma", 0.2); c.setdefault("tpe_bandwidth", 0.1); c.setdefault("tpe_eps", 1e-12)
    c.setdefault("x_dim", 1); c.setdefault("initial_n", 10); c.setdefault("rbf_variance", 1.0)
    c.setdefault("obs_noise_variance", 1.0); c.setdefault("randomized_var", 1.0)
    c.setdefault("confidence_level", 0.9); c.setdefault("enforce_complete", True); c.setdefault("max_retries_per_run", 100)
    c.setdefault("alternative", "two-sided"); c.setdefault("f_star_spec", {"type": "constant", "value": 0.0})
    d = int(c["x_dim"])
    c.setdefault("points_per_axis", points_per_axis(d))
    c.setdefault("rbf_length_scale", rbf_length(d))
    c.setdefault("selection_spec", {"window_length": list(window_length(d))})
    if "window_length" not in c["selection_spec"]:
        c["selection_spec"]["window_length"] = list(window_length(d))
    if str(c["statistic"]) != "diff":
        raise ValueError("only statistic=diff is supported")
    if str(c["acquisition"]).lower() not in {"ucb", "tpe"}:
        raise ValueError("only acquisition=ucb/tpe is supported")
    c["acquisition"] = str(c["acquisition"]).lower()
    return c


def expand_configs(config_path: Path, outdir: Optional[str] = None) -> list[dict]:
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(spec, dict):
        raise ValueError("YAML config must be a mapping")
    base = spec.get("defaults", {}) or {}
    combos: list[dict] = []
    if any(k in spec for k in ("defaults", "grid", "grids", "additions", "overrides")):
        if "grids" in spec:
            for entry in spec.get("grids") or []:
                g = entry.get("grid", entry) if isinstance(entry, dict) else {}
                combos.extend(deep_merge(base, undot(c)) for c in expand_grid(g or {}))
        else:
            combos.extend(deep_merge(base, undot(c)) for c in expand_grid(spec.get("grid", {}) or {}))
        for add in list(spec.get("additions", []) or spec.get("overrides", []) or []):
            combos.append(deep_merge(base, undot(add or {})))
    else:
        combos = [spec]
    fam = family(config_path)
    resolved, seen = [], set()
    for raw in combos:
        c = defaults(raw)
        c.setdefault("artifact_path", artifact_name(c, fam))
        if outdir:
            c["artifact_path"] = str(Path(outdir) / Path(str(c["artifact_path"])).name)
        key = yaml.safe_dump(c, sort_keys=True, allow_unicode=True)
        if key not in seen:
            seen.add(key); resolved.append(c)
    artifacts = [str(c["artifact_path"]) for c in resolved]
    if len(set(artifacts)) != len(artifacts):
        raise ValueError("duplicate artifact_path generated")
    return resolved
