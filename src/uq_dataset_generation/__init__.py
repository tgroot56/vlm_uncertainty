"""Code for generating the supervised UQ datasets used in the paper."""

from .config import SupervisionGenConfig
from .core import generate_supervised_uq_dataset

__all__ = ["SupervisionGenConfig", "generate_supervised_uq_dataset"]
