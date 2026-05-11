from __future__ import annotations

import math
from typing import Optional, cast

import numpy as np

import si_state
from si_state import get_candidates, linear_polynomials_below_zero
from si_core import (
    Node,
    SIContext,
    SIIntervalViolationError,
    IntervalMeanWindowNotFoundError,
    TrackedVector,
    interval_update,
)


class ModelBase:
    def __init__(self, noise_variance: float = 1.0):
        self.noise_variance = float(noise_variance)
        self.X: Optional[np.ndarray] = None
        self.y: Optional[TrackedVector] = None

    def set_XY(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.X is not None:
            raise RuntimeError("set_XY can be called only once")
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if x.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch")
        self.X = x.copy()
        self.y = TrackedVector(y)

    def add_XY(self, x_new: np.ndarray, y_new: float) -> None:
        if self.X is None or self.y is None:
            raise RuntimeError("call set_XY first")
        x_new = np.asarray(x_new, float)
        if x_new.ndim == 1:
            x_new = x_new[None, :]
        self.X = np.concatenate([self.X, x_new], axis=0)
        self.y.append(y_new)


class TPEModel(ModelBase):
    pass


class RBF:
    def __init__(self, length_scale: float, variance: float = 1.0):
        self.length_scale = float(length_scale)
        self.variance = float(variance)

    def __call__(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        x, z = np.asarray(x, float), np.asarray(z, float)
        d2 = np.sum((x[:, None, :] - z[None, :, :]) ** 2, axis=2)
        return self.variance * np.exp(-0.5 * d2 / self.length_scale**2)


class GPRegression(ModelBase):
    def __init__(self, kernel: RBF, noise_variance: float = 1.0):
        super().__init__(noise_variance)
        self.kernel = kernel

    def predict(self, x_test: np.ndarray, linear: bool = False):
        if self.X is None or self.y is None:
            raise RuntimeError("model has no data")
        x_test = np.asarray(x_test, float)
        if x_test.ndim == 1:
            x_test = x_test[None, :]
        k = self.kernel(self.X, self.X)
        n = k.shape[0]
        k[np.diag_indices(n)] += self.noise_variance + 1e-8
        ks = self.kernel(x_test, self.X)
        L = np.linalg.cholesky(k)
        U = np.linalg.solve(L.T, np.linalg.solve(L, ks.T))
        W = U.T
        mu = W @ self.y.value
        v = np.linalg.solve(L, ks.T)
        var = np.full(x_test.shape[0], self.kernel.variance) - np.sum(v * v, axis=0)
        std = np.sqrt(np.maximum(var, 1e-12))
        return (mu, std, W, np.zeros(x_test.shape[0])) if linear else (mu, std)


def candidate_pool(model: ModelBase) -> tuple[np.ndarray, np.ndarray]:
    if model.X is None:
        raise RuntimeError("model not initialized")
    xc = get_candidates()
    mask = np.ones(xc.shape[0], dtype=bool)
    for x in model.X:
        mask &= ~np.all(np.isclose(xc, x, atol=0, rtol=0), axis=1)
    if not np.any(mask):
        raise RuntimeError("no candidates left")
    return xc[mask], np.flatnonzero(mask)


class UCB(Node):
    def __init__(self, model: GPRegression, kappa: float):
        super().__init__()
        self.model, self.kappa = model, float(kappa)
        self.B = self.a = None
        self.n_obs = 0

    @staticmethod
    def argmax(a: np.ndarray) -> int:
        idx = np.arange(a.size)
        return int(np.lexsort((idx, -np.asarray(a, float).ravel()))[0])

    def forward(self) -> int:
        assert self.model.X is not None and self.model.y is not None
        xcand, idxmap = candidate_pool(self.model)
        _, std, B, a0 = self.model.predict(xcand, linear=True)
        self.B, self.a, self.n_obs = B, a0 + self.kappa * std, self.model.X.shape[0]
        k = self.argmax(B @ self.model.y.value + self.a)
        self.forward_model = (k,)
        return int(idxmap[k])

    def forward_si(self, ctx: SIContext) -> SIContext:
        if self.B is None or self.a is None:
            raise RuntimeError("UCB.forward must run before SI replay")
        a, b = ctx.a[: self.n_obs], ctx.b[: self.n_obs]
        scores = self.B @ (a + b * ctx.z) + self.a
        k = self.argmax(scores)
        mask = np.ones(scores.size, dtype=bool)
        mask[k] = False
        if np.any(mask):
            dB = self.B[k][None, :] - self.B[mask]
            intervals = linear_polynomials_below_zero(-(dB @ a + self.a[k] - self.a[mask]), -(dB @ b))
            ctx = interval_update(ctx, intervals, f"UCB({self.node_id})")
        self.si_model = (k,)
        return ctx


class TPE(Node):
    def __init__(self, model: TPEModel, gamma: float, bandwidth: float, eps: float):
        super().__init__()
        self.model = model
        self.gamma, self.bandwidth, self.eps = float(gamma), float(bandwidth), float(eps)
        self.n_obs = 0
        self.Xcand = None

    @staticmethod
    def top_indices(y: np.ndarray, k: int) -> np.ndarray:
        y = np.asarray(y, float).ravel()
        idx = np.arange(y.size)
        return np.sort(np.lexsort((idx, np.where(np.isnan(y), -np.inf, -y)))[:k])

    def split(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = max(1, min(int(math.ceil(self.gamma * self.n_obs)), self.n_obs))
        good = self.top_indices(y[: self.n_obs], k)
        return good, np.setdiff1d(np.arange(self.n_obs), good, assume_unique=False)

    def density(self, xq: np.ndarray, xr: np.ndarray) -> np.ndarray:
        if xr.shape[0] == 0:
            return np.zeros(xq.shape[0])
        d2 = np.sum((xq[:, None, :] - xr[None, :, :]) ** 2, axis=2)
        return np.sum(np.exp(-0.5 * d2 / self.bandwidth**2), axis=1)

    def best_candidate(self, good: np.ndarray, bad: np.ndarray) -> int:
        assert self.model.X is not None and self.Xcand is not None
        xobs = self.model.X[: self.n_obs]
        ratio = self.density(self.Xcand, xobs[good]) / (self.density(self.Xcand, xobs[bad]) + self.eps)
        return int(np.argmax(ratio))

    def forward(self) -> int:
        assert self.model.X is not None and self.model.y is not None
        self.n_obs = self.model.X.shape[0]
        self.Xcand, idxmap = candidate_pool(self.model)
        good, bad = self.split(self.model.y.value)
        k = self.best_candidate(good, bad)
        self.forward_model = (k,)
        return int(idxmap[k])

    def forward_si(self, ctx: SIContext) -> SIContext:
        a, b = ctx.a[: self.n_obs], ctx.b[: self.n_obs]
        good, bad = self.split(a + b * ctx.z)
        if good.size and bad.size:
            I = np.repeat(good, bad.size)
            J = np.tile(bad, good.size)
            intervals = linear_polynomials_below_zero(-(a[I] - a[J]), -(b[I] - b[J]))
            ctx = interval_update(ctx, intervals, f"TPE.split({self.node_id})")
        self.si_model = (self.best_candidate(good, bad),)
        return ctx


def acquire(model: ModelBase, cfg: dict) -> int:
    if str(cfg["acquisition"]).lower() == "tpe":
        node = TPE(cast(TPEModel, model), cfg["tpe_gamma"], cfg["tpe_bandwidth"], cfg["tpe_eps"])
    else:
        node = UCB(cast(GPRegression, model), cfg["kappa"])
    idx = node.forward()
    assert model.y is not None
    model.y.graph.push(node)
    return idx


class IntervalMeanWindowSelector(Node):
    def __init__(self, x_obs: np.ndarray, L: tuple[float, ...], largest: bool, exclude: Optional[np.ndarray] = None):
        super().__init__()
        x = np.asarray(x_obs, float)
        if x.ndim == 1:
            x = x[:, None]
        self.x, self.n_obs, self.largest = x, x.shape[0], bool(largest)
        self.L = np.asarray(L, float).ravel()
        if self.L.size != x.shape[1] or np.any(self.L <= 0):
            raise ValueError("window_length dimension mismatch")
        self.windows, self.W = self._make_windows(None if exclude is None else np.asarray(exclude, int))

    def _make_windows(self, exclude: Optional[np.ndarray]):
        seen, windows = set(), []
        excluded = set() if exclude is None else set(map(int, exclude.ravel()))
        for anchor in self.x:
            members = tuple(np.flatnonzero(np.all((self.x >= anchor) & (self.x - anchor <= self.L), axis=1)).tolist())
            if not members or members in seen or any(i in excluded for i in members):
                continue
            seen.add(members)
            windows.append(np.asarray(members, int))
        if not windows:
            raise IntervalMeanWindowNotFoundError("no valid interval-mean window")
        W = np.zeros((len(windows), self.n_obs))
        for k, idx in enumerate(windows):
            W[k, idx] = 1.0 / len(idx)
        return windows, W

    def best(self, scores: np.ndarray) -> int:
        idx = np.arange(scores.size)
        key = -scores if self.largest else scores
        return int(np.lexsort((idx, key))[0])

    def forward(self, y: TrackedVector) -> np.ndarray:
        k = self.best(self.W @ y.value[: self.n_obs])
        chosen = self.windows[k]
        self.forward_model = (tuple(map(int, chosen)),)
        y.graph.push(self)
        return chosen

    def forward_si(self, ctx: SIContext) -> SIContext:
        a, b = ctx.a[: self.n_obs], ctx.b[: self.n_obs]
        k = self.best(self.W @ (a + b * ctx.z))
        mask = np.ones(self.W.shape[0], dtype=bool)
        mask[k] = False
        if np.any(mask):
            dW = self.W[k][None, :] - self.W[mask]
            c, d = dW @ a, dW @ b
            intervals = linear_polynomials_below_zero(-c, -d) if self.largest else linear_polynomials_below_zero(c, d)
            ctx = interval_update(ctx, intervals, f"Window({self.node_id})")
        chosen = self.windows[k]
        self.si_model = (tuple(map(int, chosen)),)
        return ctx


def select_window(y: TrackedVector, x_obs: np.ndarray, L: tuple[float, ...], largest: bool, exclude=None) -> np.ndarray:
    return IntervalMeanWindowSelector(x_obs, L, largest, exclude).forward(y)
