
"""Common configurations."""

import dataclasses
from typing import Optional

# Import libraries

from official.core import config_definitions as cfg
from official.modeling import hyperparams

@dataclasses.dataclass
class NormActivation(hyperparams.Config):
    activation: str = 'leaky_relu'
    use_sync_bn: bool = True
    norm_momentum: float = 0.99
    norm_epsilon: float = 0.001