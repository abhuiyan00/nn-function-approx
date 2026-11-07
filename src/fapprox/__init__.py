"""fapprox - a from-scratch NumPy engine for neural function approximation.

Soft Computing, Project I, Topic 11 ("Neural network as a device for function
approximation"), Wroclaw University of Science and Technology, 2026/27.

Public API
    targets  : TARGETS, get_target
    data     : make_dataset
    models   : build_mlp, build_kan, Network
    training : TrainConfig, fit
    metrics  : evaluate, regression_metrics
"""
import os

# Before NumPy loads: its small solves gain nothing from threads, and spinning ones slow LM about 2x
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from .data import make_dataset  # noqa: E402
from .functions import TARGETS, get_target  # noqa: E402
from .metrics import evaluate, regression_metrics  # noqa: E402
from .model import Network, build_kan, build_mlp  # noqa: E402
from .train import TrainConfig, fit  # noqa: E402

__all__ = ["TARGETS", "get_target", "make_dataset", "Network", "build_mlp", "build_kan",
           "TrainConfig", "fit", "evaluate", "regression_metrics"]
__version__ = "0.1.0"
