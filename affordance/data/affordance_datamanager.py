from dataclasses import dataclass, field
from lerf.data.lerf_datamanager import LERFDataManager, LERFDataManagerConfig
import torch
from typing import Literal, Tuple, Type, Dict, Union
from affordance.data.utils.affordance_dataloader import AffordanceDataLoader
from pathlib import Path
import os.path as osp
from nerfstudio.cameras.rays import RayBundle
import numpy as np
from nerfstudio.utils.rich_utils import CONSOLE

@dataclass
class AffordanceDataManagerConfig(LERFDataManagerConfig):
    _target: Type = field(default_factory=lambda:AffordanceDataManager)

class AffordanceDataManager(LERFDataManager):
    def __init__(
            self,
            config: AffordanceDataManagerConfig,
            device: Union[torch.device, str] = "cpu",
            test_mode: Literal['test', 'val', 'inference'] = "val",
            world_size: int = 1,
            local_rank: int = 0, 
            **kwargs):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs) 
        images = [self.train_dataset[i]['image'].permute(2,0,1)[None, ...] for i in range(len(self.train_dataset))]
        images = torch.cat(images)
        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        locate_cache_path = Path(osp.join(cache_dir, "affordance/locate.npy"))
        self.affordance_dataloader = AffordanceDataLoader(
            image_list=images,
            device=self.device,
            cfg={"image_shape": list(images.shape[2:4])},
            cache_path=locate_cache_path,
        )
        CONSOLE.print('inside train')


        

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        batch["clip"], clip_scale = self.clip_interpolator(ray_indices)
        batch["dino"] = self.dino_dataloader(ray_indices)
        batch["locate"] = self.affordance_dataloader(ray_indices)
        ray_bundle.metadata["clip_scales"] = clip_scale
        # assume all cameras have the same focal length and image width
        ray_bundle.metadata["fx"] = self.train_dataset.cameras[0].fx.item()
        ray_bundle.metadata["width"] = self.train_dataset.cameras[0].width.item()
        ray_bundle.metadata["fy"] = self.train_dataset.cameras[0].fy.item()
        ray_bundle.metadata["height"] = self.train_dataset.cameras[0].height.item()
        return ray_bundle, batch

