"""AIORI 2.0 federated learning: eight algorithms on one single-process engine. See README.md."""

from aiori2.fl.algorithms import REGISTRY, weighted_sum
from aiori2.fl.engine import FL_ARMS, FederatedRun, local_train, make_config, validate

ALGORITHMS = tuple(REGISTRY)
__all__ = ["ALGORITHMS", "FL_ARMS", "FederatedRun", "REGISTRY", "local_train", "make_config", "validate", "weighted_sum"]
