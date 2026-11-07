"""Request bodies. Limits exist so one click cannot exhaust the machine."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fapprox.functions import TARGETS
from fapprox.model import kan_param_count, mlp_param_count

MAX_PARAMS = 200_000
MAX_PARAMS_LM = 1_500  # LM solves a P x P system every iteration
Width = Annotated[int, Field(ge=1, le=512)]
Seed = Annotated[int, Field(ge=0, le=2 ** 31 - 1)]
MODEL_NAME = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"  # one path component, never hidden


class RunSpec(BaseModel):
    """One training run; fields mirror ``python -m fapprox train``."""
    model_config = ConfigDict(extra="forbid")

    target: str = "f1"
    arch: Literal["mlp", "kan"] = "mlp"
    hidden: list[Width] = Field(default=[32, 32], min_length=1, max_length=6)
    act: Literal["tanh", "sigmoid", "relu", "silu", "sine"] = "tanh"
    bias_init: Literal["zero", "kink"] | None = None
    fourier_m: int = Field(0, ge=0, le=512)
    fourier_scale: float = Field(2.0, gt=0, le=128)
    w0: float = Field(30.0, gt=0, le=300)
    widths: list[Annotated[int, Field(ge=1, le=64)]] | None = Field(None, min_length=2, max_length=5)
    grid: int = Field(5, ge=2, le=64)
    grid_updates: list[Annotated[int, Field(ge=1)]] = Field(default=[], max_length=8)
    optimizer: Literal["adam", "sgd", "momentum", "nesterov", "lm"] = "adam"
    lr: float = Field(1e-3, gt=0, le=10)
    batch: int = Field(64, ge=0, le=20_000)
    epochs: int = Field(300, ge=1, le=50_000)
    patience: int = Field(50, ge=0, le=50_000)
    l2: float = Field(0.0, ge=0, le=10)
    n_train: int = Field(2000, ge=10, le=20_000)
    n_val: int = Field(500, ge=5, le=5_000)
    sigma: float = Field(0.0, ge=0, le=5)
    seed: Seed = 1
    data_seed: Seed = 0

    @model_validator(mode="after")
    def _consistent(self):
        target = TARGETS.get(self.target)
        if target is None:
            raise ValueError(f"unknown target {self.target!r}; choose from {sorted(TARGETS)}")
        if self.arch == "kan":
            self.widths = self.widths or [target.dim, 5, 1]
            if self.widths[0] != target.dim:
                raise ValueError(f"KAN widths must start with the input dimension {target.dim}")
            if self.widths[-1] != 1:
                raise ValueError("KAN widths must end with 1: the engine has scalar output")
        elif self.grid_updates:
            raise ValueError("grid updates apply to KAN networks only")
        n = self.param_count(target.dim)
        if n > MAX_PARAMS:
            raise ValueError(f"{n:,} parameters exceed the limit of {MAX_PARAMS:,}")
        if self.optimizer == "lm" and n > MAX_PARAMS_LM:
            raise ValueError(f"Levenberg-Marquardt solves a P × P system each iteration: "
                             f"{n:,} parameters exceed its limit of {MAX_PARAMS_LM:,}")
        return self

    def param_count(self, dim: int) -> int:
        if self.arch == "kan":
            return kan_param_count(self.widths, self.grid)
        return mlp_param_count(dim, self.hidden, fourier_m=self.fourier_m)

    def describe(self) -> str:
        opt = {"adam": "Adam", "sgd": "SGD", "momentum": "Momentum", "nesterov": "Nesterov", "lm": "LM"}[self.optimizer]
        if self.arch == "kan":
            return f"KAN [{','.join(map(str, self.widths))}] G={self.grid} · {opt}"
        same = len(set(self.hidden)) == 1
        shape = f"{len(self.hidden)}×{self.hidden[0]}" if same else "-".join(map(str, self.hidden))
        name = f"SIREN {shape} ω₀={self.w0:g}" if self.act == "sine" else f"MLP {shape} {self.act}"
        if self.fourier_m:
            name = f"Fourier {self.fourier_m} + {name}"
        return f"{name} · {opt}"


class SaveRequest(BaseModel):
    name: str = Field(pattern=MODEL_NAME)


class PredictRequest(BaseModel):
    inputs: list[list[float]] = Field(min_length=1, max_length=10_000)


class JobRequest(BaseModel):
    experiment: str
    profile: Literal["quick", "full"] = "quick"
    workers: int = Field(2, ge=1, le=64)
