from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

import si_state
from si_state import RealSubset


class SIIntervalViolationError(Exception):
    pass


class IntervalMeanWindowNotFoundError(Exception):
    pass


class RandomizedFlagOffError(Exception):
    pass


class InverseCDFNaNError(Exception):
    pass


@dataclass(frozen=True)
class SelectionSpec:
    window_length: tuple[float, ...]


@dataclass(frozen=True)
class SIContext:
    a: np.ndarray
    b: np.ndarray
    z: float
    interval: RealSubset


class Graph:
    def __init__(self):
        self.nodes: list[Node] = []

    def push(self, node: "Node") -> None:
        self.nodes.append(node)

    def reset(self) -> None:
        for node in self.nodes:
            node.si_model = None

    def algorithm(self, a: np.ndarray, b: np.ndarray, z: float):
        ctx = SIContext(np.asarray(a, float), np.asarray(b, float), float(z), RealSubset([[-np.inf, np.inf]]))
        for node in self.nodes:
            ctx = node.forward_si(ctx)
        model = tuple(node.si_model for node in self.nodes if node.forward_model is not None or node.si_model is not None)
        return model, ctx.interval

    def model_selector(self, model: tuple) -> bool:
        obs = tuple(node.forward_model for node in self.nodes if node.forward_model is not None)
        return tuple(model) == obs


class Node:
    _next_id = 0

    def __init__(self):
        self.node_id = Node._next_id
        Node._next_id += 1
        self.forward_model = None
        self.si_model = None

    def forward_si(self, ctx: SIContext) -> SIContext:
        raise NotImplementedError


def interval_update(ctx: SIContext, intervals, label: str, tol: float = 1e-8) -> SIContext:
    intervals = list(intervals)
    if not intervals:
        raise SIIntervalViolationError(f"{label}: empty interval")
    lo, hi = map(float, intervals[0])
    z = float(ctx.z)
    if z < lo:
        if lo - z <= tol:
            lo = float(np.nextafter(z, -np.inf))
        else:
            raise SIIntervalViolationError(f"{label}: z={z} below [{lo}, {hi}]")
    if z > hi:
        if z - hi <= tol:
            hi = float(np.nextafter(z, np.inf))
        else:
            raise SIIntervalViolationError(f"{label}: z={z} above [{lo}, {hi}]")
    new_interval = ctx.interval & RealSubset([[lo, hi]])
    if new_interval.is_empty():
        raise SIIntervalViolationError(f"{label}: empty intersection at z={z}")
    return SIContext(ctx.a, ctx.b, ctx.z, new_interval)


class TrackedVector:
    def __init__(self, y: np.ndarray):
        self.pure = np.asarray(y, float).ravel().copy()
        self.randomizer: Optional[np.ndarray] = None
        if si_state._RANDOMIZED:
            if si_state._RANDOMIZED_VAR is None:
                raise ValueError("randomized_var is required")
            self.randomizer = si_state._RNG.normal(0.0, math.sqrt(si_state._RANDOMIZED_VAR), size=self.pure.shape)
            self.value = self.pure + self.randomizer
        else:
            self.value = self.pure.copy()
        self.graph = Graph()

    def append(self, y_new: float) -> None:
        y_new = float(np.asarray(y_new, float).reshape(-1)[0])
        self.pure = np.append(self.pure, y_new)
        if self.randomizer is None:
            self.value = np.append(self.value, y_new)
        else:
            assert si_state._RANDOMIZED_VAR is not None
            w = float(si_state._RNG.normal(0.0, math.sqrt(si_state._RANDOMIZED_VAR)))
            self.randomizer = np.append(self.randomizer, w)
            self.value = np.append(self.value, y_new + w)
