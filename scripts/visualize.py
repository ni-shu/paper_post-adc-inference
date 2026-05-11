#!/usr/bin/env python3
from __future__ import annotations

import glob, itertools, sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

METHODS = ["naive", "randomized", "randomized_path_only", "randomized_selector_only", "bonferroni"]
STYLE = {
    "randomized_path_only": (10, "#ff7f0e", "s", "--", r"w/o-$\eta$"),
    "randomized_selector_only": (20, "#2ca02c", "^", ":", r"w/o-$\mathcal{T}$"),
    "bonferroni": (30, "#d62728", "D", "--", "Bonferroni"),
    "naive": (40, "#7f7f7f", "X", "-.", "naive"),
    "randomized": (50, "#1f77b4", "o", "-", "post-ADC"),
}
BASE = dict(
    alpha=.05, methods=METHODS, plot_methods=None, dpi=160,
    equal_x=False, strict_x=True, figsize=(4, 4), fs=10, label_fs=10,
    legend_fs=9, xtick_fs=9, ytick_fs=9, lw=2.0, ms=8.0, legend_ncol=1,
    y_label=None, y_lim=None, x_lim=None, y_scale=None, stagger=False,
    filter={}, x_list=None, sweep=None, alpha_line=True,
)

def cfg(**kw):
    x = dict(BASE); x.update(kw); return x

PRESETS = {
    "power": cfg(
        glob="runs/synthetic_power__*.parquet", outdir="./plots/power",
        x_axis="f_star_spec.a", x_label=r"Signal amplitude $a$", x_list=[1.0, 2.0, 4.0, 8.0],
        sweep={"acquisition": ["ucb", "tpe"], "f_star_spec.type": ["sinc", "cosine", "chirp", "gaussian_bump", "nonsmooth_peak", "negative_forrester"]},
        methods=["randomized", "bonferroni"], plot_methods=["randomized", "bonferroni"],
        equal_x=True, figsize=(4.0, 2.8), label_fs=12, legend_ncol=2, lw=.8, ms=5.0, y_lim=(-.05, 1.05),
        plots=[("is_sig", "pr", "power", dict(alpha_line=False, y_label="Power"))],
    ),
    "fpr_hyperparameter": cfg(
        glob="runs/synthetic_hyper_*.parquet", outdir="./plots/fpr_hyperparameter", x_axis="kappa", strict_x=False, xtick_fs=8,
        plots=[
            ("is_sig", "pr", "hyper_ucb", dict(x_label=r"Exploration parameter $\kappa$", x_list=[.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], equal_x=True, stagger=True, y_label="Type I error rate", filter={"acquisition": "ucb"})),
            ("is_sig", "pr", "hyper_tpe", dict(x_axis="tpe_gamma", x_label=r"Quantile level $\gamma$", x_list=[.1, .2, .3, .4, .5, .6, .7, .8], y_label="Type I error rate", filter={"acquisition": "tpe"})),
        ],
    ),
    "fpr_nsteps": cfg(
        glob="runs/synthetic_fpr_nsteps__*.parquet", outdir="./plots/fpr_nsteps",
        x_axis="num_steps", x_label=r"Number of steps $N_{\mathrm{steps}}$", x_list=[25, 50, 75, 100],
        sweep={"acquisition": ["tpe", "ucb"]}, figsize=(4.2, 2.8), alpha_line=False,
        plots=[
            ("is_sig", "pr", "fpr", dict(alpha_line=True, y_label="Type I error rate", label_fs=12, y_lim=(-.05, 1.05), x_lim=(15, 110), figsize=(3.2, 2.8), lw=.5, ms=5.0)),
            ("coverage", "pr", "coverage", dict(y_label="Coverage rate", label_fs=12, y_lim=(-.05, 1.05), x_lim=(15, 110), figsize=(3.2, 2.8), lw=.5, ms=5.0)),
            ("ci_len", "box", "ci_len", dict(y_label="Confidence interval length", y_scale="log", y_lim=(.9, 120), methods=["naive", "randomized", "bonferroni"], plot_methods=["naive", "randomized", "bonferroni"])),
        ],
    ),
    "fpr_dim": cfg(
        glob="runs/synthetic_fpr_dim__*.parquet", outdir="./plots/fpr_dim",
        x_axis="x_dim", x_label=r"Problem dimension $d$", x_list=[1, 2, 3], sweep={"acquisition": ["tpe", "ucb"]},
        plots=[("is_sig", "pr", "fpr_dim", dict(y_label="Type I error rate"))],
    ),
}

def merge(a, b):
    x = dict(a); x.update(b or {}); return x

def load(pattern):
    ps = sorted(glob.glob(pattern))
    if not ps: raise FileNotFoundError(pattern)
    print(f"[INFO] {pattern}: {len(ps)} file(s)")
    return pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)

def combos(sweep):
    if not sweep: return [{}]
    ks = list(sweep)
    return [dict(zip(ks, vs)) for vs in itertools.product(*(sweep[k] for k in ks))]

def filter_rows(df, spec):
    for k, v in (spec or {}).items():
        if k not in df: raise KeyError(k)
        df = df[df[k].isin(v) if isinstance(v, (list, tuple, set)) else df[k].eq(v)]
    return df

def metric_frame(raw, c, combo, metric):
    d = filter_rows(raw, {**c.get("filter", {}), **combo})
    d = d[d.method.isin(c["methods"])].copy()
    if d.empty: return pd.DataFrame(columns=["x", "method", "value"])
    p, lo, hi = d.p_value.astype(float), d.ci_low.astype(float), d.ci_high.astype(float)
    th = d.theta_true.astype(float)
    vals = dict(ci_len=hi - lo, is_sig=(p < c["alpha"]).astype(float), coverage=((lo <= th) & (th <= hi)).astype(float))[metric]
    out = pd.DataFrame({"x": d[c["x_axis"]], "method": d.method.astype(str), "value": vals}).dropna()
    if c.get("x_lim"):
        x = pd.to_numeric(out.x, errors="raise")
        out = out[(x >= c["x_lim"][0]) & (x <= c["x_lim"][1])]
    if c.get("x_list"):
        present = set(out.x.tolist())
        keep = [x for x in c["x_list"] if x in present]
        missing = [x for x in c["x_list"] if x not in present]
        if missing and c.get("strict_x", True): raise ValueError(f"missing x: {missing}")
        if missing: print(f"[WARN] skipped missing x: {missing}")
        out = out[out.x.isin(keep)].copy()
        out["x"] = pd.Categorical(out.x, categories=keep, ordered=True)
    return out

def ordered_methods(df, c):
    ms = list(dict.fromkeys(df.method.astype(str)))
    if c.get("plot_methods"):
        ms = [m for m in c["plot_methods"] if m in ms]
    return sorted(ms, key=lambda m: STYLE[m][0])

def as_num(xs):
    try: return pd.to_numeric(pd.Series(xs), errors="raise").to_numpy(float)
    except Exception: return None

def style(m, c):
    _, color, marker, ls, _ = STYLE[m]
    return dict(color=color, marker=marker, linestyle=ls, linewidth=c["lw"], markersize=c["ms"], markeredgewidth=.8, markeredgecolor="white")

def label(m): return STYLE[m][4]

def start(c):
    plt.rcParams.update({"axes.labelsize": c["label_fs"], "axes.titlesize": c["fs"], "legend.fontsize": c["legend_fs"], "xtick.labelsize": c["xtick_fs"], "ytick.labelsize": c["ytick_fs"]})
    plt.figure(figsize=c["figsize"])

def finish(pdf, c, metric, title):
    ax = plt.gca()
    ax.set(xlabel=c["x_label"], ylabel=c.get("y_label") or metric, title=title)
    ax.title.set_y(1.02)
    if c.get("y_scale"): ax.set_yscale(c["y_scale"])
    if c.get("x_lim"): ax.set_xlim(*c["x_lim"])
    if c.get("y_lim"): ax.set_ylim(*c["y_lim"])
    if c.get("stagger"):
        for i, t in enumerate(ax.get_xticklabels()):
            t.set_y(-.03 - .06 * (i % 2)); t.set_verticalalignment("top")
    ax.grid(True, linestyle=":", alpha=.6)
    ax.legend(loc="best", ncol=c.get("legend_ncol", 1))
    plt.tight_layout(); pdf.savefig(plt.gcf()); plt.close()

def wilson(k, n, z=1.959963984540054):
    if n <= 0: return np.nan, np.nan
    p = k / n; den = 1 + z*z/n
    mid = (p + z*z/(2*n)) / den
    half = z * np.sqrt((p*(1-p) + z*z/(4*n)) / n) / den
    return max(0.0, mid-half), min(1.0, mid+half)

def pr(pdf, d, c, metric, suffix):
    start(c); ax = plt.gca()
    ms = ordered_methods(d, c)
    for m in ms:
        a = d[d.method.eq(m)].groupby("x", observed=True).value.agg(k="sum", n="count").reset_index().sort_values("x")
        xs, xn = a.x.tolist(), as_num(a.x.tolist())
        X = np.arange(len(xs)) if c.get("equal_x") or xn is None else xn
        p = (a.k / a.n).to_numpy(float)
        ci = [wilson(round(k), round(n)) for k, n in zip(a.k, a.n)]
        yerr = np.array([[max(0, y-lo), max(0, hi-y)] for y, (lo, hi) in zip(p, ci)]).T
        ax.errorbar(X, p, yerr=yerr, ecolor=STYLE[m][1], capsize=3, label=label(m), **style(m, c))
        ax.set_xticks(X, [str(x) for x in xs])
    if metric == "is_sig" and c.get("alpha_line"):
        ax.axhline(c["alpha"], color="black", linestyle=(0, (6, 3)), linewidth=1.8, zorder=0, clip_on=False, label=f"alpha={c['alpha']:.2f}")
    if not c.get("y_lim"): ax.set_ylim(0, 1)
    finish(pdf, c, metric, suffix)

def box(pdf, d, c, metric, suffix):
    start(c); ax = plt.gca()
    xs = list(dict.fromkeys(sorted(d.x.tolist(), key=lambda x: (isinstance(x, str), x))))
    Xn = as_num(xs); base = np.arange(len(xs), dtype=float) if c.get("equal_x") or Xn is None else Xn
    ms = ordered_methods(d, c); width = ((base[1] - base[0]) if len(base) > 1 else 1) * .8; dodge = width / (len(ms) + 1)
    for i, m in enumerate(ms):
        arrays, pos = [], []
        for x, b in zip(xs, base):
            y = d[d.x.eq(x) & d.method.eq(m)].value.dropna().to_numpy()
            if len(y): arrays.append(y); pos.append(b - width/2 + (i+1)*dodge)
        if not arrays: continue
        bp = ax.boxplot(arrays, positions=pos, widths=dodge*.9, manage_ticks=False, patch_artist=True)
        for patch in bp["boxes"]: patch.set(facecolor=STYLE[m][1], edgecolor="black", alpha=.6)
        for key in ("whiskers", "caps", "medians", "fliers"):
            for line in bp.get(key, []): line.set(color="black")
        ax.plot([], [], label=label(m), **style(m, c))
    ax.set_xticks(base, [str(x) for x in xs])
    finish(pdf, c, metric, suffix)

def draw_empty(pdf, title):
    plt.figure(); plt.title(title); plt.axis("off"); pdf.savefig(plt.gcf()); plt.close()

def run(name):
    base = PRESETS[name]; raw = load(base["glob"]); outdir = Path(base["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    for metric, kind, stem, over in base["plots"]:
        c = merge(base, over); path = outdir / f"{stem}.pdf"
        with PdfPages(path) as pdf:
            for combo in combos(c.get("sweep")):
                d = metric_frame(raw, c, combo, metric)
                def _fmt(k, v):
                    if k == "acquisition": return f"ADC={v}"
                    if k == "f_star_spec.type": return f"f*={v}"
                    return f"{k}={v}"
                suffix = ", ".join(_fmt(k, v) for k, v in combo.items())
                if d.empty: draw_empty(pdf, f"No data: {metric} | {suffix}")
                elif kind == "pr": pr(pdf, d, c, metric, suffix)
                else: box(pdf, d, c, metric, suffix)
        print(f"[INFO] wrote {path}")

def main():
    names = sys.argv[1:] or ["all"]
    for name in list(PRESETS) if "all" in names else names:
        if name not in PRESETS: raise SystemExit(f"choose one of: {', '.join(PRESETS)} or all")
        run(name)

if __name__ == "__main__":
    main()
