from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from sicore import RealSubset, RandomizedSelectiveInference, linear_polynomials_below_zero
    from sicore.core.real_subset import NotBelongToSubsetError
    from sicore.main.inference import InverseCDFNaNError as SICoreInverseCDFNaNError
except ModuleNotFoundError as exc:
    _SICORE_IMPORT_ERROR = exc
    RandomizedSelectiveInference = None
    SICoreInverseCDFNaNError = RuntimeError

    class NotBelongToSubsetError(Exception):
        pass

    class RealSubset:
        def __init__(self, intervals):
            self.intervals = [(float(a), float(b)) for a, b in intervals if float(a) <= float(b)]

        def __and__(self, other):
            out = []
            for a, b in self.intervals:
                for c, d in other.intervals:
                    lo, hi = max(a, c), min(b, d)
                    if lo <= hi:
                        out.append((lo, hi))
            return RealSubset(out)

        def is_empty(self):
            return not self.intervals

        def __iter__(self):
            return iter(self.intervals)

    def linear_polynomials_below_zero(*_, **__):
        raise _SICORE_IMPORT_ERROR


RANDOMIZED_METHODS = {"randomized", "randomized_path_only", "randomized_selector_only"}
ALL_DEFAULT_METHODS = ["naive", "randomized", "randomized_path_only", "randomized_selector_only", "bonferroni"]

_CANDIDATES: Optional[np.ndarray] = None
_RANDOMIZED = False
_RANDOMIZED_VAR: Optional[float] = None
_RNG = np.random.default_rng()


def set_candidates(x: np.ndarray) -> None:
    global _CANDIDATES
    x = np.asarray(x, float)
    if x.ndim != 2:
        raise ValueError("candidates must be a 2D array")
    _CANDIDATES = x.copy()


def get_candidates() -> np.ndarray:
    if _CANDIDATES is None:
        raise RuntimeError("candidates are not set")
    return _CANDIDATES


def set_randomized(flag: bool, randomized_var: Optional[float] = None) -> None:
    global _RANDOMIZED, _RANDOMIZED_VAR
    _RANDOMIZED = bool(flag)
    _RANDOMIZED_VAR = None if randomized_var is None else float(randomized_var)


def set_seed(seed: int) -> None:
    global _RNG
    _RNG = np.random.default_rng(int(seed))


def next_seed() -> int:
    return int(_RNG.integers(0, np.iinfo(np.uint32).max))
