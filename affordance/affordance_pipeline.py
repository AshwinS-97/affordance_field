"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from affordance.data.affordance_datamanager import AffordanceDataManagerConfig
from affordance.affordance_model import TemplateModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from lerf.lerf_pipeline import LERFPipelineConfig, LERFPipeline


@dataclass
class TemplatePipelineConfig(LERFPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: TemplatePipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = AffordanceDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = TemplateModelConfig()
    """specifies the model config"""


class TemplatePipeline(LERFPipeline):
    """Template Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """


    def __init__(
        self,
        config: TemplatePipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config=config, device=device)