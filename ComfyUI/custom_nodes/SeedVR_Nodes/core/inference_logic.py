# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import List, Optional, Tuple, Union
import torch
from einops import rearrange
from omegaconf import DictConfig, ListConfig
from torch import Tensor

# Adjusted imports
from .config_utils import create_object, load_config # Added load_config

# TODO: Address these imports by copying relevant files or refactoring
from .common_utils.decorators import log_on_entry, log_runtime # Assuming decorators.py will be in common_utils
# from common.decorators import log_on_entry, log_runtime

# Assuming these will be in common_utils/diffusion_utils.py or similar
from .common_utils.diffusion_utils import (
    classifier_free_guidance_dispatcher,
    create_sampler_from_config,
    create_sampling_timesteps_from_config,
    create_schedule_from_config,
)
# from common.diffusion import (
#     classifier_free_guidance_dispatcher,
#     create_sampler_from_config,
#     create_sampling_timesteps_from_config,
#     create_schedule_from_config,
# )

from .common_utils.distributed_ops import get_device # Using the one from common_utils
from .common_utils.distributed_advanced import get_global_rank # Using the one from common_utils (placeholder for actual global rank)
from .common_utils.dist_meta_utils import meta_non_persistent_buffer_init_fn # Assuming dist_meta_utils.py

# from common.fs import download # Appears unused in this specific class, can be removed if not needed

from .dit_v2_models import na # Use 'na' from the copied dit_v2_models

# Removed placeholder NAUtils class and instance
# Removed placeholder get_device, get_global_rank, meta_non_persistent_buffer_init_fn

# Data transform imports (may be unused in this class directly but added as per instruction)
from .data_transforms.divisible_crop import DivisibleCrop
from .data_transforms.na_resize import NaResize
from .data_transforms.rearrange import Rearrange

class VideoDiffusionInfer():
    def __init__(self, config: DictConfig):
        self.config = config

    def get_condition(self, latent: Tensor, latent_blur: Tensor, task: str) -> Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if task == "t2v" or t == 1:
            # t2i or t2v generation.
            if task == "sr":
                cond[:, ..., :-1] = latent_blur[:]
                cond[:, ..., -1:] = 1.0
            return cond
        if task == "i2v":
            # i2v generation.
            cond[:1, ..., :-1] = latent[:1]
            cond[:1, ..., -1:] = 1.0
            return cond
        if task == "v2v":
            # v2v frame extension.
            cond[:2, ..., :-1] = latent[:2]
            cond[:2, ..., -1:] = 1.0
            return cond
        if task == "sr":
            # sr generation.
            cond[:, ..., :-1] = latent_blur[:]
            cond[:, ..., -1:] = 1.0
            return cond
        raise NotImplementedError

    # @log_on_entry # Removed for now
    # @log_runtime # Removed for now
    def configure_dit_model(self, device="cpu", checkpoint=None):
        # Load dit checkpoint.
        # For fast init & resume,
        #   when training from scratch, rank0 init DiT on cpu, then sync to other ranks with FSDP.
        #   otherwise, all ranks init DiT on meta device, then load_state_dict with assign=True.
        if self.config.dit.get("init_with_meta_device", False):
            init_device = "cpu" if get_global_rank() == 0 and checkpoint is None else "meta"
        else:
            init_device = "cpu"

        # Create dit model.
        with torch.device(init_device):
            # TODO: This 'create_object' relies on 'self.config.dit.model' which has paths like 'models.dit_v2.DiTMoE_v2_3B_patch2_145_cond'
            # These paths will need to be updated or handled by a custom import_item in config_utils for the new structure.
            self.dit = create_object(self.config.dit.model)
        self.dit.set_gradient_checkpointing(self.config.dit.gradient_checkpoint)

        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", mmap=True)
            loading_info = self.dit.load_state_dict(state, strict=True, assign=True)
            print(f"Loading pretrained ckpt from {checkpoint}")
            print(f"Loading info: {loading_info}")
            self.dit = meta_non_persistent_buffer_init_fn(self.dit)

        if device in [get_device(), "cuda"]: # Simplified device check
            self.dit.to(get_device())

        # Print model size.
        num_params = sum(p.numel() for p in self.dit.parameters() if p.requires_grad)
        print(f"DiT trainable parameters: {num_params:,}")

    # @log_on_entry # Removed for now
    # @log_runtime # Removed for now
    def configure_vae_model(self):
        # Create vae model.
        dtype = getattr(torch, self.config.vae.dtype)
        # TODO: This 'create_object' relies on 'self.config.vae.model' which has paths like 'models.vae.VQGANVAE_TATS_Spatial_v2'
        # These paths will need to be updated or handled.
        self.vae = create_object(self.config.vae.model)
        self.vae.requires_grad_(False).eval()
        self.vae.to(device=get_device(), dtype=dtype)

        # Load vae checkpoint.
        state = torch.load(
            self.config.vae.checkpoint, map_location=get_device(), mmap=True
        )
        self.vae.load_state_dict(state)

        # Set causal slicing.
        if hasattr(self.vae, "set_causal_slicing") and hasattr(self.config.vae, "slicing"):
            self.vae.set_causal_slicing(**self.config.vae.slicing)

    # ------------------------------ Diffusion ------------------------------ #

    def configure_diffusion(self):
        # TODO: These create_..._from_config functions are from common.diffusion and need to be available.
        self.schedule = create_schedule_from_config(
            config=self.config.diffusion.schedule,
            device=get_device(),
        )
        self.sampling_timesteps = create_sampling_timesteps_from_config(
            config=self.config.diffusion.timesteps.sampling,
            schedule=self.schedule,
            device=get_device(),
        )
        self.sampler = create_sampler_from_config(
            config=self.config.diffusion.sampler,
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
        )

    # -------------------------------- Helper ------------------------------- #

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor]) -> List[Tensor]:
        use_sample = self.config.vae.get("use_sample", True)
        latents = []
        if len(samples) > 0:
            device = get_device()
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group samples of the same shape to batches if enabled.
            if self.config.vae.grouping:
                batches, indices = na.pack(samples) # Uses placeholder 'na'
            else:
                batches = [sample.unsqueeze(0) for sample in samples]

            # Vae process by each group.
            for sample in batches:
                sample = sample.to(device, dtype)
                if hasattr(self.vae, "preprocess"):
                    sample = self.vae.preprocess(sample)
                if use_sample:
                    latent = self.vae.encode(sample).latent
                else:
                    # Deterministic vae encode, only used for i2v inference (optionally)
                    latent = self.vae.encode(sample).posterior.mode().squeeze(2)
                latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
                latent = rearrange(latent, "b c ... -> b ... c")
                latent = (latent - shift) * scale
                latents.append(latent)

            # Ungroup back to individual latent with the original order.
            if self.config.vae.grouping:
                latents = na.unpack(latents, indices) # Uses placeholder 'na'
            else:
                latents = [latent.squeeze(0) for latent in latents]

        return latents

    @torch.no_grad()
    def vae_decode(self, latents: List[Tensor]) -> List[Tensor]:
        samples = []
        if len(latents) > 0:
            device = get_device()
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group latents of the same shape to batches if enabled.
            if self.config.vae.grouping:
                latents, indices = na.pack(latents) # Uses placeholder 'na'
            else:
                latents = [latent.unsqueeze(0) for latent in latents]

            # Vae process by each group.
            for latent in latents:
                latent = latent.to(device, dtype)
                latent = latent / scale + shift
                latent = rearrange(latent, "b ... c -> b c ...")
                latent = latent.squeeze(2)
                sample = self.vae.decode(latent).sample
                if hasattr(self.vae, "postprocess"):
                    sample = self.vae.postprocess(sample)
                samples.append(sample)

            # Ungroup back to individual sample with the original order.
            if self.config.vae.grouping:
                samples = na.unpack(samples, indices) # Uses placeholder 'na'
            else:
                samples = [sample.squeeze(0) for sample in samples]

        return samples

    def timestep_transform(self, timesteps: Tensor, latents_shapes: Tensor):
        # Skip if not needed.
        if not self.config.diffusion.timesteps.get("transform", False):
            return timesteps

        # Compute resolution.
        vt = self.config.vae.model.get("temporal_downsample_factor", 4)
        vs = self.config.vae.model.get("spatial_downsample_factor", 8)
        frames = (latents_shapes[:, 0] - 1) * vt + 1
        heights = latents_shapes[:, 1] * vs
        widths = latents_shapes[:, 2] * vs

        # Compute shift factor.
        def get_lin_function(x1, y1, x2, y2):
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return lambda x: m * x + b

        img_shift_fn = get_lin_function(x1=256 * 256, y1=1.0, x2=1024 * 1024, y2=3.2)
        vid_shift_fn = get_lin_function(x1=256 * 256 * 37, y1=1.0, x2=1280 * 720 * 145, y2=5.0)
        shift = torch.where(
            frames > 1,
            vid_shift_fn(heights * widths * frames),
            img_shift_fn(heights * widths),
        )

        # Shift timesteps.
        timesteps = timesteps / self.schedule.T
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * self.schedule.T
        return timesteps

    @torch.no_grad()
    def inference(
        self,
        noises: List[Tensor],
        conditions: List[Tensor],
        texts_pos: Union[List[str], List[Tensor], List[Tuple[Tensor]]],
        texts_neg: Union[List[str], List[Tensor], List[Tuple[Tensor]]],
        cfg_scale: Optional[float] = None,
        dit_offload: bool = False,
    ) -> List[Tensor]:
        assert len(noises) == len(conditions) == len(texts_pos) == len(texts_neg)
        batch_size = len(noises)

        # Return if empty.
        if batch_size == 0:
            return []

        # Set cfg scale
        if cfg_scale is None:
            cfg_scale = self.config.diffusion.cfg.scale

        # Text embeddings.
        assert type(texts_pos[0]) is type(texts_neg[0])
        if isinstance(texts_pos[0], str):
            # TODO: self.text_encode is not defined in this class. It's likely part of the original larger script or class.
            # This will need to be implemented or brought in.
            # For now, this will raise an AttributeError.
            text_pos_embeds, text_pos_shapes = self.text_encode(texts_pos)
            text_neg_embeds, text_neg_shapes = self.text_encode(texts_neg)
        elif isinstance(texts_pos[0], tuple):
            text_pos_embeds, text_pos_shapes = [], []
            text_neg_embeds, text_neg_shapes = [], []
            for pos in zip(*texts_pos):
                emb, shape = na.flatten(pos) # Uses placeholder 'na'
                text_pos_embeds.append(emb)
                text_pos_shapes.append(shape)
            for neg in zip(*texts_neg):
                emb, shape = na.flatten(neg) # Uses placeholder 'na'
                text_neg_embeds.append(emb)
                text_neg_shapes.append(shape)
        else:
            text_pos_embeds, text_pos_shapes = na.flatten(texts_pos) # Uses placeholder 'na'
            text_neg_embeds, text_neg_shapes = na.flatten(texts_neg) # Uses placeholder 'na'

        # Flatten.
        latents, latents_shapes = na.flatten(noises) # Uses placeholder 'na'
        latents_cond, _ = na.flatten(conditions) # Uses placeholder 'na'

        # Enter eval mode.
        was_training = self.dit.training
        self.dit.eval()

        # Sampling.
        # TODO: classifier_free_guidance_dispatcher is from common.diffusion and needs to be available.
        latents = self.sampler.sample(
            x=latents,
            f=lambda args: classifier_free_guidance_dispatcher(
                pos=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_pos_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_pos_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                neg=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_neg_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_neg_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                scale=(
                    cfg_scale
                    if (args.i + 1) / len(self.sampler.timesteps)
                    <= self.config.diffusion.cfg.get("partial", 1)
                    else 1.0
                ),
                rescale=self.config.diffusion.cfg.rescale,
            ),
        )

        # Exit eval mode.
        self.dit.train(was_training)

        # Unflatten.
        latents = na.unflatten(latents, latents_shapes) # Uses placeholder 'na'

        if dit_offload:
            self.dit.to("cpu")

        # Vae decode.
        self.vae.to(get_device())
        samples = self.vae_decode(latents)

        if dit_offload:
            self.dit.to(get_device())
        return samples
