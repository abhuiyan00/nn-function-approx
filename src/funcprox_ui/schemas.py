"""Request bodies. Limits exist so one click cannot exhaust the machine."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from funcprox.custom import make_target
from funcprox.functions import Target
from funcprox.model import kan_param_count, mlp_param_count

MAX_PARAMS = 200_000
MAX_PARAMS_LM = 1_500  # LM solves a P x P system every iteration
MAX_POINTS = 5_000
Width = Annotated[int, Field(ge=1, le=512)]
Seed = Annotated[int, Field(ge=0, le=2 ** 31 - 1)]
Range = tuple[float, float]
MODEL_NAME = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"  # one path component, never hidden


class FunctionSpec(BaseModel):
    """What to approximate: a built-in target, a formula or points, with an optional missing middle."""
    model_config = ConfigDict(extra="forbid")

    target: str = "f1"
    formula: str | None = None
    domain: list[Range] | None = Field(None, max_length=3)
    points: list[Range] | None = Field(None, max_length=MAX_POINTS)
    gap: list[Range] | None = Field(None, max_length=3)  # null keeps the target's own hole, empty removes it
    _made: Target = PrivateAttr()

    @model_validator(mode="after")
    def _resolve(self):
        self._made = make_target(**self.function_fields())
        self.target = self._made.key
        return self

    def function_fields(self) -> dict:
        fields = dict(target=self.target, formula=self.formula, domain=self.domain, points=self.points, gap=self.gap)
        return {k: v for k, v in fields.items() if v is not None}

    @property
    def resolved(self) -> Target:
        return self._made


class RunSpec(FunctionSpec):
    """One training run; fields mirror ``python -m funcprox train``."""

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
        target = self.resolved
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


class SearchRequest(BaseModel):
    """Find the closest curve: ``base`` gives the function and the data, the panel of models is fixed."""
    model_config = ConfigDict(extra="forbid")

    base: RunSpec
    effort: Literal["quick", "thorough"] = "quick"
    seeds: int = Field(1, ge=1, le=5)


class SweepRequest(BaseModel):
    """A sensitivity study: train ``base`` once per value of one setting, several seeds each."""
    model_config = ConfigDict(extra="forbid")

    base: RunSpec
    param: Literal["lr", "batch", "l2", "n_train", "sigma", "patience", "epochs", "width", "depth", "grid"]
    values: list[float] = Field(min_length=2, max_length=12)
    seeds: int = Field(3, ge=1, le=5)
    _specs: list = PrivateAttr(default_factory=list)

    def spec_for(self, value: float, seed: int) -> RunSpec:
        """``base`` with the swept setting at ``value``; width and depth reshape the hidden layers."""
        fields = self.base.model_dump()
        if self.param == "width":
            fields["hidden"] = [int(value)] * len(self.base.hidden)
        elif self.param == "depth":
            fields["hidden"] = [self.base.hidden[0]] * int(value)
        else:
            fields[self.param] = value
        return RunSpec(**{**fields, "seed": seed})

    @model_validator(mode="after")
    def _values_fit(self):
        if len(set(self.values)) != len(self.values):
            raise ValueError("every value must differ")
        if self.param in ("width", "depth", "grid") and (self.param == "grid") != (self.base.arch == "kan"):
            raise ValueError("width and depth apply to MLPs, grid to KANs")
        self._specs = [[self.spec_for(v, seed) for seed in range(1, self.seeds + 1)] for v in sorted(self.values)]
        return self


class JobRequest(BaseModel):
    experiment: str
    profile: Literal["quick", "full"] = "quick"
    workers: int = Field(2, ge=1, le=64)
