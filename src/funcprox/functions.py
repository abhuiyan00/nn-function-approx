"""Target functions (the "unknown devices" the network must imitate).

Each target is a known closed-form function, so the trained network can be scored
against the *exact* function on a dense test grid, not only against noisy labels.
Every target was chosen to probe one property of neural approximation.

| key | formula | domain | property probed |
|-----|---------|--------|-----------------|
| f1  | sin(2 pi x)                         | [-1, 1]   | smooth, periodic: reference case |
| f2  | 2 x^2                               | [-2, 2]   | extrapolation (tested on [-3, 3]) |
| f3  | sin(pi x) + 0.5 sin(4 pi x) + 0.25 sin(16 pi x) | [-1, 1] | multi-scale: spectral bias |
| f4  | 1 / (1 + 25 x^2)                    | [-1, 1]   | Runge function: polynomial failure |
| f5  | Franke (1979) function              | [0, 1]^2  | classic 2-D benchmark |
| f6  | abs(x)                              | [-1, 1]   | non-smooth kink: ReLU vs tanh |
| f7  | exp(sin(pi x) + y^2)                | [-1, 1]^2 | compositional: KAN paper toy example |
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Target:
    key: str
    name: str
    formula: str
    dim: int
    domain: tuple  # ((lo, hi),) per input dimension
    fn: Callable[[Array], Array]
    probes: str
    extrapolation_domain: tuple | None = None
    # integer k of components sin(k pi x) for spectral analysis (f3 only)
    frequencies: tuple = field(default_factory=tuple)

    def __call__(self, x: Array) -> Array:
        x = np.atleast_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(f"{self.key} expects {self.dim} input columns, got {x.shape[1]}")
        return self.fn(x).reshape(-1, 1)


def _franke(x: Array) -> Array:
    u, v = x[:, 0], x[:, 1]
    return (
        0.75 * np.exp(-((9 * u - 2) ** 2) / 4 - ((9 * v - 2) ** 2) / 4)
        + 0.75 * np.exp(-((9 * u + 1) ** 2) / 49 - (9 * v + 1) / 10)
        + 0.5 * np.exp(-((9 * u - 7) ** 2) / 4 - ((9 * v - 3) ** 2) / 4)
        - 0.2 * np.exp(-((9 * u - 4) ** 2) - (9 * v - 7) ** 2)
    )


TARGETS: dict[str, Target] = {
    "f1": Target("f1", "sine", "sin(2πx)", 1, ((-1.0, 1.0),),
                 lambda x: np.sin(2 * np.pi * x[:, 0]),
                 "smooth periodic reference"),
    "f2": Target("f2", "quadratic", "2x²", 1, ((-2.0, 2.0),),
                 lambda x: 2.0 * x[:, 0] ** 2,
                 "extrapolation outside the training box",
                 extrapolation_domain=((-3.0, 3.0),)),
    "f3": Target("f3", "multiscale", "sin(πx) + 0.5 sin(4πx) + 0.25 sin(16πx)", 1, ((-1.0, 1.0),),
                 lambda x: (np.sin(np.pi * x[:, 0]) + 0.5 * np.sin(4 * np.pi * x[:, 0])
                            + 0.25 * np.sin(16 * np.pi * x[:, 0])),
                 "spectral bias (low frequencies are learned first)",
                 frequencies=(1, 4, 16)),
    "f4": Target("f4", "runge", "1 / (1 + 25x²)", 1, ((-1.0, 1.0),),
                 lambda x: 1.0 / (1.0 + 25.0 * x[:, 0] ** 2),
                 "Runge phenomenon of polynomial interpolation"),
    "f5": Target("f5", "franke", "Franke's function", 2, ((0.0, 1.0), (0.0, 1.0)),
                 _franke, "classic 2-D scattered-data benchmark"),
    "f6": Target("f6", "kink", "|x|", 1, ((-1.0, 1.0),),
                 lambda x: np.abs(x[:, 0]),
                 "non-differentiable point"),
    "f7": Target("f7", "compositional", "exp(sin(πx) + y²)", 2, ((-1.0, 1.0), (-1.0, 1.0)),
                 lambda x: np.exp(np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2),
                 "compositional structure (Kolmogorov-Arnold form)"),
}


def get_target(key: str) -> Target:
    try:
        return TARGETS[key]
    except KeyError as exc:
        raise KeyError(f"unknown target {key!r}; choose from {sorted(TARGETS)}") from exc
