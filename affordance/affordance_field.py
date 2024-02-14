"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Literal, Optional, Dict
from jaxtyping import Float
from torch import Tensor

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field
from lerf.lerf_field import LERFField
from nerfstudio.field_components.spatial_distortions import SceneContraction
import torch
from nerfstudio.cameras.rays import RaySamples
import tinycudann as tcnn
from enum import Enum
from lerf.lerf_fieldheadnames import LERFFieldHeadNames
# class LERFFieldHeadNames(Enum):
#     """Possible field outputs"""
#     HASHGRID = "hashgrid"
#     CLIP = "clip"
#     DINO = "dino"
#     # AFFORDANCES = "affordances"

class TemplateNerfField(LERFField):
    """Template Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    aabb: Tensor

    def __init__(
        self,
        grid_layers,
        grid_sizes,
        grid_resolutions,
        clip_n_dims: int,
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ) -> None:
        super().__init__(grid_layers, grid_sizes, grid_resolutions, clip_n_dims)
        tot_out_dims = sum([e.n_output_dims for e in self.clip_encs])
        self.affordance_net = tcnn.Network(
            n_input_dims=tot_out_dims,
            n_output_dims=36,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 1,
            },
        )
    def get_outputs(self, ray_samples: RaySamples, clip_scales) -> Dict[LERFFieldHeadNames, Float[Tensor, "bs dim"]]:
        # random scales, one scale
        outputs = {}

        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 3)) for e in self.clip_encs]
        x = torch.concat(xs, dim=-1)

        outputs[LERFFieldHeadNames.HASHGRID] = x.view(*ray_samples.frustums.shape, -1)

        clip_pass = self.clip_net(torch.cat([x, clip_scales.view(-1, 1)], dim=-1)).view(*ray_samples.frustums.shape, -1)
        outputs[LERFFieldHeadNames.CLIP] = clip_pass / clip_pass.norm(dim=-1, keepdim=True)
        # assert False
        dino_pass = self.dino_net(x).view(*ray_samples.frustums.shape, -1)
        outputs[LERFFieldHeadNames.DINO] = dino_pass

        affordance_pass = self.affordance_net(x).view(*ray_samples.frustums.shape, -1)
        outputs['affordance'] = affordance_pass
        return outputs


    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.
