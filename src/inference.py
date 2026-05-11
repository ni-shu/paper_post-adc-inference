from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np
from scipy.stats import norm

import si_state
from si_state import NotBelongToSubsetError, SICoreInverseCDFNaNError
from si_core import (
    Graph,
    TrackedVector,
    SIIntervalViolationError,
    InverseCDFNaNError,
    RandomizedFlagOffError,
)
from bo_models import IntervalMeanWindowSelector


class MeanDiff:
    def __init__(self, y: TrackedVector, I: np.ndarray, J: np.ndarray):
        I, J = np.asarray(I, int), np.asarray(J, int)
        if I.size == 0 or J.size == 0 or np.intersect1d(I, J).size:
            raise ValueError("I and J must be non-empty and disjoint")
        self.y, self.I, self.J = y, I, J

    def eta(self) -> np.ndarray:
        out = np.zeros(self.y.pure.size)
        out[self.I] = 1.0 / self.I.size
        out[self.J] = -1.0 / self.J.size
        return out

    def true_value(self, truth_on_observed: np.ndarray) -> float:
        return float(self.eta() @ truth_on_observed)


def ablated_stat(stat: MeanDiff, method: str) -> MeanDiff:
    keep_selector = method == "randomized_selector_only"
    y2 = copy.copy(stat.y)
    y2.graph = Graph()
    for node in stat.y.graph.nodes:
        is_selector = isinstance(node, IntervalMeanWindowSelector)
        if is_selector == keep_selector:
            y2.graph.nodes.append(copy.copy(node))
    return MeanDiff(y2, stat.I.copy(), stat.J.copy())


def ztest(z: float, var: float, alternative: str, correction_log_m: Optional[float] = None):
    scale = math.sqrt(max(float(var), 1e-300))
    zs = z / scale
    if alternative == "greater":
        logp = float(norm.logsf(zs))
    elif alternative == "less":
        logp = float(norm.logcdf(zs))
    else:
        logp = float(math.log(2.0) + norm.logsf(abs(zs)))
    if correction_log_m is not None:
        logp = min(0.0, logp + correction_log_m)

    class Result:
        method = "bonferroni" if correction_log_m is not None else "naive"
        p_value = float(math.exp(logp))
        mle = float(z)

        def ci(self, confidence_level: float = 0.95):
            if correction_log_m is None:
                q = float(norm.isf((1.0 - confidence_level) / 2.0))
            else:
                log_tail = math.log(1.0 - confidence_level) - correction_log_m - math.log(2.0)
                try:
                    from scipy import special
                    q = float(-special.ndtri_exp(log_tail))
                except Exception:
                    q = float(norm.isf(math.exp(log_tail)))
            return z - q * scale, z + q * scale

    return Result()


def infer(stat: MeanDiff, method: str, var: float, alternative: str, log_m: Optional[float] = None):
    y, eta = stat.y.pure, stat.eta()
    var_stat = float(var) * float(eta @ eta)
    z_obs = float(eta @ y)
    if method == "naive":
        return ztest(z_obs, var_stat, alternative)
    if method == "bonferroni":
        return ztest(z_obs, var_stat, alternative, log_m)
    if method != "randomized":
        raise ValueError(f"unsupported method: {method}")
    if not si_state._RANDOMIZED:
        raise RandomizedFlagOffError("randomized flag is off")
    if si_state.RandomizedSelectiveInference is None or stat.y.randomizer is None or si_state._RANDOMIZED_VAR is None:
        raise RuntimeError("randomized inference requires sicore and stored randomizer")
    scale = math.sqrt(var_stat)
    si = si_state.RandomizedSelectiveInference(
        data=y,
        var=float(var),
        randomizer=stat.y.randomizer.astype(float),
        randomized_var=float(si_state._RANDOMIZED_VAR),
        eta=eta / scale,
        alternative={"greater": "less", "less": "greater", "two-sided": "two-sided"}[alternative],
        null_value=0.0,
    )
    graph = stat.y.graph
    graph.reset()

    class Result:
        method = "randomized"

        def __init__(self):
            self.scale = scale
            self.alternative = alternative
            self.default_level = 0.9 if alternative == "two-sided" else 0.8
            self.res = self._run(confidence_level=self.default_level)
            self.p_value = self.res.p_value

        def _run(self, **kwargs):
            try:
                return si.inference(algorithm=graph.algorithm, model_selector=graph.model_selector, **kwargs)
            except NotBelongToSubsetError as e:
                raise SIIntervalViolationError(str(e)) from e

        def ci(self, confidence_level: float = 0.9):
            one_sided = self.alternative in {"greater", "less"}
            level = 2.0 * confidence_level - 1.0 if one_sided else confidence_level
            if abs(level - self.default_level) <= 1e-12 and self.res.confidence_interval is not None:
                ci = self.res.confidence_interval
            else:
                graph.reset()
                ci = self._run(inference_mode="parametric", confidence_level=level).confidence_interval
            if ci is None:
                return float("nan"), float("nan")
            lo, hi = self.scale * float(ci[0]), self.scale * float(ci[1])
            if not one_sided:
                return lo, hi
            return (lo, float("inf")) if self.alternative == "greater" else (float("-inf"), hi)

        @property
        def mle(self):
            pe = self.res.point_estimate
            return self.scale * (float(pe) if pe is not None else float("nan"))

    return Result()


def summarize(res, confidence_level: float) -> dict:
    try:
        lo, hi = res.ci(confidence_level=confidence_level)
    except Exception as e:
        if "InverseCDFNaNError" in str(e):
            raise InverseCDFNaNError(str(e))
        raise
    return {"p_value": float(res.p_value), "mle": float(getattr(res, "mle", np.nan)), "ci90": (float(lo), float(hi))}
