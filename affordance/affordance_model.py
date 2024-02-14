"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type, Dict
import numpy as np
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.viewer.viewer_elements import ViewerCheckbox, ViewerButton, ViewerText
from lerf.lerf import LERFModel, LERFModelConfig
import torch.nn as nn
import torch
from collections import defaultdict 
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
import gc
from affordance.affordance_field import TemplateNerfField
from nerfstudio.cameras.rays import RayBundle, RaySamples
from lerf.lerf_fieldheadnames import LERFFieldHeadNames
@dataclass
class TemplateModelConfig(LERFModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: TemplateModel)


class TemplateModel(LERFModel):
    """Template Model."""

    config: TemplateModelConfig
      


    def populate_modules(self):
        super().populate_modules()
        # self.save_out = save_output()
        self.lerf_field = TemplateNerfField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            clip_n_dims=self.image_encoder.embedding_dim,
        )

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples)
        lerf_weights, best_ids = torch.topk(weights, self.config.num_lerf_samples, dim=-2, sorted=False)

        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        lerf_samples: RaySamples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

        if self.training:
            with torch.no_grad():
                clip_scales = ray_bundle.metadata["clip_scales"]
                clip_scales = clip_scales[..., None]
                dist = (lerf_samples.frustums.get_positions() - ray_bundle.origins[:, None, :]).norm(
                    dim=-1, keepdim=True
                )
            clip_scales = clip_scales * ray_bundle.metadata["height"] * (dist / ray_bundle.metadata["fy"])
        else:
            clip_scales = torch.ones_like(lerf_samples.spacing_starts, device=self.device)

        override_scales = (
            None if "override_scales" not in ray_bundle.metadata else ray_bundle.metadata["override_scales"]
        )
        weights_list.append(weights)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        lerf_field_outputs = self.lerf_field.get_outputs(lerf_samples, clip_scales)

        outputs["clip"] = self.renderer_clip(
            embeds=lerf_field_outputs[LERFFieldHeadNames.CLIP], weights=lerf_weights.detach()
        )
        outputs["dino"] = self.renderer_mean(
            embeds=lerf_field_outputs[LERFFieldHeadNames.DINO], weights=lerf_weights.detach()
        )
        outputs["affordance"] = self.renderer_mean(
            embeds=lerf_field_outputs["affordance"], weights=lerf_weights.detach()
        )
        if not self.training:
            with torch.no_grad():
                max_across, best_scales = self.get_max_across(
                    lerf_samples,
                    lerf_weights,
                    lerf_field_outputs[LERFFieldHeadNames.HASHGRID],
                    clip_scales.shape,
                    preset_scales=override_scales,
                )
                outputs["raw_relevancy"] = max_across  # N x B x 1
                outputs["best_scales"] = best_scales.to(self.device)  # N

        return outputs
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            unreduced_clip = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
                outputs["clip"], batch["clip"], delta=1.25, reduction="none"
            )
            loss_dict["clip_loss"] = unreduced_clip.sum(dim=-1).nanmean()
            unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
            loss_dict["dino_loss"] = unreduced_dino.sum(dim=-1).nanmean()
            unreduced_aff = torch.nn.functional.mse_loss(outputs["affordance"], batch["locate"], reduction="none")
            loss_dict["affordance_loss"] = unreduced_aff.sum(dim=-1).nanmean()

        return loss_dict


    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
            """Takes in camera parameters and computes the output of the model.

            LERF overrides this from base_model since we need to compute the max_across relevancy in multiple batches,
            which are not independent since they need to use the same scale
            Args:
                camera_ray_bundle: ray bundle to calculate outputs over
            """
            # TODO(justin) implement max across behavior
            num_rays_per_chunk = self.config.eval_num_rays_per_chunk
            image_height, image_width = camera_ray_bundle.origins.shape[:2]
            num_rays = len(camera_ray_bundle)
            outputs_lists = defaultdict(list)  # dict from name:list of outputs (1 per bundle)
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                outputs = self.forward(ray_bundle=ray_bundle)
                # take the best scale for each query across each ray bundle
                if i == 0:
                    best_scales = outputs["best_scales"]
                    best_relevancies = [m.max() for m in outputs["raw_relevancy"]]
                else:
                    for phrase_i in range(outputs["best_scales"].shape[0]):
                        m = outputs["raw_relevancy"][phrase_i, ...].max()
                        if m > best_relevancies[phrase_i]:
                            best_scales[phrase_i] = outputs["best_scales"][phrase_i]
                            best_relevancies[phrase_i] = m
            # re-render the max_across outputs using the best scales across all batches
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                ray_bundle.metadata["override_scales"] = best_scales
                outputs = self.forward(ray_bundle=ray_bundle)
                # standard nerfstudio concatting
                for output_name, output in outputs.items():  # type: ignore
                    if output_name == "best_scales":
                        continue
                    if output_name == "raw_relevancy":
                        for r_id in range(output.shape[0]):
                            outputs_lists[f"relevancy_{r_id}"].append(output[r_id, ...])
                    else:
                        outputs_lists[output_name].append(output)
            outputs = {}
            for output_name, outputs_list in outputs_lists.items():
                if not torch.is_tensor(outputs_list[0]):
                    # TODO: handle lists of tensors as well
                    continue
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
            for i in range(len(self.image_encoder.positives)):
                p_i = torch.clip(outputs[f"relevancy_{i}"] - 0.5, 0, 1)
                # a_i = torch.clip(outputs['affordance'], 0, 1)
                outputs[f"composited_{i}"] = apply_colormap(p_i / (p_i.max() + 1e-6), ColormapOptions("turbo"))
                # outputs[f"com_aff_{i}"] = apply_colormap(a_i[..., 17:18] / (a_i[..., 17].max() + 1e-6), ColormapOptions("turbo"))
                mask = (outputs["relevancy_0"] < 0.5).squeeze()
                # mask_inv = (outputs["relevancy_0"] >= 0.5).squeeze()
                outputs[f"composited_{i}"][mask, :] = outputs["rgb"][mask, :]
                # outputs[f"com_aff_{i}"] = apply_colormap(a_i[..., 5:6], ColormapOptions("turbo"))
                # outputs[f"com_aff_{i}"] = outputs['affordance']
                # outputs[f"com_aff_{i}"][mask, :] = outputs["rgb"][mask, :]
                # outputs[f"com_aff_{i}"][mask_inv, :] = outputs["rgb"][mask, :]
            return outputs

# class save_output(nn.Module):#must inherit from nn.Module
#     def __init__(self):
#         # Must be a class variable
#         super().__init__()
#         self.a = ViewerButton(name="Update_Text",cb_hook=self.updateText)
#         self.f = ViewerText(name="Text", default_value="Hello World")
#     def updateText(self, handle: ViewerButton) -> None:
#         for obj in gc.get_objects():
#             if isinstance(obj, TemplateModel):
#                 self.f.name = obj

        