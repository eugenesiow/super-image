"""
super-image package.

A library of image super resolution algorithms in PyTorch. For upscaling images.
"""

from .configuration_utils import PretrainedConfig
from .modeling_utils import PreTrainedModel
from .trainer import Trainer
from .training_args import TrainingArguments

from .models import (
    EdsrModel,
    EdsrConfig,
    MsrnModel,
    MsrnConfig,
    A2nModel,
    A2nConfig,
    PanModel,
    PanConfig,
    MasaModel,
    MasaConfig,
    CarnModel,
    CarnConfig
)

from .data import (
    ImageLoader
)

from typing import List

__all__: List[str] = ['TrainingArguments', 'Trainer',
                      'EdsrModel', 'EdsrConfig',
                      'MsrnModel', 'MsrnConfig',
                      'A2nModel', 'A2nConfig',
                      'PanModel', 'PanConfig',
                      'MasaModel', 'MasaConfig',
                      'CarnModel', 'CarnConfig',
                      'ImageLoader']  # noqa: WPS410 (the only __variable__ we use)
