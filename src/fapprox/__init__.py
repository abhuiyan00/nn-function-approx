"""fapprox - a hand-written NumPy engine for neural function approximation.

Soft Computing, Project I, Topic 11 ("Neural network as a device for function
approximation"), Wroclaw University of Science and Technology, 2026/27.

Public API
    targets  : TARGETS, get_target
    data     : make_dataset
    models   : build_mlp, build_kan, Network
    training : TrainConfig, fit
    metrics  : evaluate, regression_metrics
"""
from .data import make_dataset
from .functions import TARGETS, get_target
from .metrics import evaluate, regression_metrics
from .model import Network, build_kan, build_mlp
from .train import TrainConfig, fit

__all__ = ["TARGETS", "get_target", "make_dataset", "Network", "build_mlp", "build_kan",
           "TrainConfig", "fit", "evaluate", "regression_metrics"]
__version__ = "0.1.0"
