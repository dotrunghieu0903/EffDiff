#!/usr/bin/env python3
"""
Knowledge Distillation module for Stable Diffusion optimization.

Implements state-of-the-art distillation methods based on:

1. DMD2 (Improved Distribution Matching Distillation) — Yin et al., NeurIPS 2024
   - Distribution matching loss with fake/real score estimation
   - GAN-style adversarial loss on clean latents
   - Two-timescale update rule for stable training

2. SANA-Sprint (Continuous-Time Consistency Distillation) — Chen et al., 2025
   - CTCD: Continuous-time consistency distillation via JVP-based tangent estimation
   - LADD: Latent adversarial diffusion distillation with multi-scale discriminator
   - Combined CTCD + LADD for one-step high-quality generation

Supported model families: SDXL, Flux, SD3, SANA
"""

import os
import sys
import gc
import json
import time
import math
import argparse
import copy
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# ============================================================================
#  Compatibility patches (referenced from quantization/quantized.py)
# ============================================================================

# Fix compatibility issues with newer versions of dependencies
try:
    from transformers.modeling_utils import apply_chunking_to_forward
except ImportError:
    try:
        from transformers.utils import apply_chunking_to_forward
    except ImportError:
        def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
            return forward_fn(*input_tensors)
        import transformers.modeling_utils
        transformers.modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward

# Fix huggingface_hub compatibility issues
try:
    from huggingface_hub.constants import HF_HOME
except ImportError:
    try:
        from huggingface_hub import constants
        if not hasattr(constants, 'HF_HOME'):
            constants.HF_HOME = os.path.expanduser("~/.cache/huggingface")
            import huggingface_hub.constants
            huggingface_hub.constants.HF_HOME = constants.HF_HOME
    except Exception as e:
        print(f"Warning: Could not set up HF_HOME compatibility: {e}")

# Fix DDUFEntry import issue
try:
    from huggingface_hub import DDUFEntry
except ImportError:
    class DDUFEntry:
        def __init__(self, *args, **kwargs):
            pass
    import huggingface_hub
    huggingface_hub.DDUFEntry = DDUFEntry

# Fix missing functions in huggingface_hub for transformers/diffusers compatibility
import huggingface_hub
try:
    from huggingface_hub import split_torch_state_dict_into_shards
    huggingface_hub.split_torch_state_dict_into_shards = split_torch_state_dict_into_shards
except ImportError:
    def split_torch_state_dict_into_shards(state_dict, *args, **kwargs):
        return {}, {'pytorch_model.bin': list(state_dict.keys())}
    huggingface_hub.split_torch_state_dict_into_shards = split_torch_state_dict_into_shards

try:
    import transformers.modeling_utils
    transformers.modeling_utils.split_torch_state_dict_into_shards = huggingface_hub.split_torch_state_dict_into_shards
except ImportError:
    pass

try:
    import transformers.generation.utils
    transformers.generation.utils.split_torch_state_dict_into_shards = huggingface_hub.split_torch_state_dict_into_shards
except ImportError:
    pass

try:
    import transformers
    if not hasattr(transformers, 'split_torch_state_dict_into_shards'):
        transformers.split_torch_state_dict_into_shards = huggingface_hub.split_torch_state_dict_into_shards
except ImportError:
    pass

# Fix missing read_dduf_file function in huggingface_hub
try:
    from huggingface_hub import read_dduf_file
    huggingface_hub.read_dduf_file = read_dduf_file
except ImportError:
    def read_dduf_file(file_path, *args, **kwargs):
        with open(file_path, 'rb') as f:
            return f.read()
    huggingface_hub.read_dduf_file = read_dduf_file

# Patch at sys.modules level
if 'transformers.generation.utils' in sys.modules:
    sys.modules['transformers.generation.utils'].split_torch_state_dict_into_shards = huggingface_hub.split_torch_state_dict_into_shards
if 'transformers.modeling_utils' in sys.modules:
    sys.modules['transformers.modeling_utils'].split_torch_state_dict_into_shards = huggingface_hub.split_torch_state_dict_into_shards

# Fix cached_download function name change
try:
    from huggingface_hub import cached_download
except ImportError:
    try:
        from huggingface_hub import hf_hub_download as cached_download
        huggingface_hub.cached_download = cached_download
    except ImportError:
        pass

# ============================================================================
#  End compatibility patches
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

# Add parent directory to path for importing shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Memory optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _import_shared():
    """Lazy import of shared modules that have heavy dependencies (pynvml, etc.)."""
    from shared.cleanup import setup_memory_optimizations
    from shared.resources_monitor import generate_image_and_monitor, write_generation_metadata_to_file
    from shared.metrics import (
        calculate_fid_subset, compute_image_reward,
        calculate_clip_score, calculate_lpips, calculate_psnr_resized
    )
    from shared.resizing_image import resize_images
    from dataset.coco import process_coco, process_coco_extended
    from dataset.flickr8k import process_flickr8k
    return {
        "setup_memory_optimizations": setup_memory_optimizations,
        "generate_image_and_monitor": generate_image_and_monitor,
        "write_generation_metadata_to_file": write_generation_metadata_to_file,
        "calculate_fid_subset": calculate_fid_subset,
        "compute_image_reward": compute_image_reward,
        "calculate_clip_score": calculate_clip_score,
        "calculate_lpips": calculate_lpips,
        "calculate_psnr_resized": calculate_psnr_resized,
        "resize_images": resize_images,
        "process_coco": process_coco,
        "process_coco_extended": process_coco_extended,
        "process_flickr8k": process_flickr8k,
    }


# ============================================================================
#  DMD2: Distribution Matching Distillation 2
#  (Yin et al., "Improved Distribution Matching Distillation", NeurIPS 2024)
# ============================================================================

class FakeScoreNetwork(nn.Module):
    """
    Fake score estimator for DMD2.

    A copy of the teacher UNet/Transformer trained to estimate the score
    of the *generated* distribution. This is contrasted against the real
    score (teacher) to produce the distribution matching gradient.

    The same network doubles as a GAN discriminator when cls_on_clean_image
    is enabled — a small classification head is appended to the bottleneck
    features.
    """

    def __init__(self, teacher_denoiser: nn.Module, model_type: str, device: str,
                 deepcopy_fn=None):
        super().__init__()
        if deepcopy_fn is not None:
            self.fake_denoiser = deepcopy_fn(teacher_denoiser)
        else:
            self.fake_denoiser = copy.deepcopy(teacher_denoiser)
        self.fake_denoiser.requires_grad_(True)
        self.model_type = model_type

        # Build a lightweight classification head on the bottleneck features
        # for GAN-style real/fake discrimination
        bottleneck_dim = self._get_bottleneck_dim()
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        ).to(device)
        self.cls_head.requires_grad_(True)

    def _get_bottleneck_dim(self) -> int:
        """Estimate bottleneck channel dimension from the denoiser architecture."""
        if self.model_type == "sdxl":
            return 1280
        elif self.model_type in ("flux", "sd3"):
            return 1536
        elif self.model_type == "sana":
            return 2240
        return 1280

    def score_forward(self, latents, timesteps, encoder_hidden_states, **kwargs):
        """Forward pass for score estimation (standard denoising prediction)."""
        output = self.fake_denoiser(latents, timesteps, encoder_hidden_states=encoder_hidden_states, **kwargs)
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output

    def classify(self, latents: torch.Tensor) -> torch.Tensor:
        """Classify latents as real or fake using the appended head."""
        return self.cls_head(latents)


class DMD2Loss(nn.Module):
    """
    Distribution Matching Distillation loss (DMD2).

    Core idea: match the *distribution* of generated samples to real data
    by comparing two denoising score estimates:
      - s_real: teacher's score on noised generated samples (real score)
      - s_fake: fake-score network's score on the same samples

    Generator gradient:
        grad = (p_real - p_fake) / mean(|p_real|)
        where p = x - denoise(x)

    This is combined with a GAN loss where the fake-score network also acts
    as a discriminator on clean latents.
    """

    def __init__(
        self,
        dm_loss_weight: float = 1.0,
        gan_loss_weight: float = 1.0,
        real_guidance_scale: float = 6.0,
        fake_guidance_scale: float = 1.0,
    ):
        super().__init__()
        self.dm_loss_weight = dm_loss_weight
        self.gan_loss_weight = gan_loss_weight
        self.real_guidance_scale = real_guidance_scale
        self.fake_guidance_scale = fake_guidance_scale

    def compute_distribution_matching_loss(
        self,
        generated_latents: torch.Tensor,
        teacher_denoiser: nn.Module,
        fake_score_net: FakeScoreNetwork,
        text_embedding: torch.Tensor,
        scheduler,
        alphas_cumprod: torch.Tensor,
        num_train_timesteps: int,
        model_type: str,
        device: str,
        forward_kwargs: dict,
        min_step_pct: float = 0.02,
        max_step_pct: float = 0.98,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the distribution matching gradient for generator update."""
        batch_size = generated_latents.shape[0]
        min_step = int(min_step_pct * num_train_timesteps)
        max_step = int(max_step_pct * num_train_timesteps)

        with torch.no_grad():
            timesteps = torch.randint(
                min_step, min(max_step + 1, num_train_timesteps),
                [batch_size], device=device, dtype=torch.long,
            )
            noise = torch.randn_like(generated_latents)
            noisy_latents = scheduler.add_noise(generated_latents, noise, timesteps)

            # Real score (teacher)
            pred_real_noise = teacher_denoiser(
                noisy_latents, timesteps,
                encoder_hidden_states=text_embedding,
                **forward_kwargs,
            )
            if isinstance(pred_real_noise, tuple):
                pred_real_noise = pred_real_noise[0]
            elif hasattr(pred_real_noise, "sample"):
                pred_real_noise = pred_real_noise.sample

            pred_real_x0 = _get_x0_from_noise(
                noisy_latents.double(), pred_real_noise.double(),
                alphas_cumprod.double(), timesteps,
            )

            # Fake score
            pred_fake_noise = fake_score_net.score_forward(
                noisy_latents, timesteps,
                encoder_hidden_states=text_embedding,
                **forward_kwargs,
            )

            pred_fake_x0 = _get_x0_from_noise(
                noisy_latents.double(), pred_fake_noise.double(),
                alphas_cumprod.double(), timesteps,
            )

            p_real = (generated_latents.double() - pred_real_x0)
            p_fake = (generated_latents.double() - pred_fake_x0)

            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True).clamp(min=1e-8)
            grad = torch.nan_to_num(grad)

        # Pseudo-loss that produces the correct gradient
        loss = 0.5 * F.mse_loss(
            generated_latents.float(),
            (generated_latents - grad).detach().float(),
        )

        metrics = {
            "loss_dm": loss.item(),
            "dm_grad_norm": torch.norm(grad).item(),
        }
        return loss * self.dm_loss_weight, metrics

    def compute_gan_loss_generator(
        self,
        generated_latents: torch.Tensor,
        fake_score_net: FakeScoreNetwork,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """GAN loss for the generator (fool the discriminator)."""
        pred_fake = fake_score_net.classify(generated_latents)
        loss = F.softplus(-pred_fake).mean()
        return loss * self.gan_loss_weight, {"gen_cls_loss": loss.item()}

    def compute_fake_score_loss(
        self,
        generated_latents: torch.Tensor,
        fake_score_net: FakeScoreNetwork,
        scheduler,
        text_embedding: torch.Tensor,
        num_train_timesteps: int,
        device: str,
        forward_kwargs: dict,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Train the fake score network to estimate score on generated data."""
        generated_latents = generated_latents.detach()
        batch_size = generated_latents.shape[0]
        noise = torch.randn_like(generated_latents)
        timesteps = torch.randint(0, num_train_timesteps, [batch_size], device=device, dtype=torch.long)
        noisy_latents = scheduler.add_noise(generated_latents, noise, timesteps)

        pred_noise = fake_score_net.score_forward(
            noisy_latents, timesteps,
            encoder_hidden_states=text_embedding,
            **forward_kwargs,
        )
        loss = F.mse_loss(pred_noise.float(), noise.float())
        return loss, {"loss_fake_score": loss.item()}

    def compute_gan_loss_discriminator(
        self,
        real_latents: torch.Tensor,
        fake_latents: torch.Tensor,
        fake_score_net: FakeScoreNetwork,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """GAN loss for the discriminator (distinguish real from fake)."""
        pred_real = fake_score_net.classify(real_latents.detach())
        pred_fake = fake_score_net.classify(fake_latents.detach())
        loss = F.softplus(pred_fake).mean() + F.softplus(-pred_real).mean()
        return loss, {
            "guidance_cls_loss": loss.item(),
            "pred_real_mean": torch.sigmoid(pred_real).mean().item(),
            "pred_fake_mean": torch.sigmoid(pred_fake).mean().item(),
        }


# ============================================================================
#  SANA-Sprint: CTCD + LADD
#  (Chen et al., "SANA-Sprint: One-Step Diffusion with Continuous-Time
#   Consistency Distillation", 2025)
# ============================================================================

class CTCDLoss(nn.Module):
    """
    Continuous-Time Consistency Distillation (CTCD) loss.

    Uses the trigonometric flow formulation:
        x_t = cos(t) * x_0 + sin(t) * z * sigma_data

    The loss enforces consistency along the ODE trajectory via
    JVP (Jacobian-Vector Product) based tangent estimation:

        g = -cos^2(t) * (sigma_data * F_theta_stop - dxdt)
            - r * (cos(t)*sin(t)*x_t + sigma_data * F_theta_grad_jvp)

    With tangent normalization: g := g / (||g|| + c)

    Final loss:
        L = (weight / exp(logvar)) * ||F_theta - F_theta_stop - g||^2  + logvar
    """

    def __init__(
        self,
        sigma_data: float = 1.0,
        tangent_warmup_steps: int = 1000,
        tangent_norm_constant: float = 0.1,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.tangent_warmup_steps = tangent_warmup_steps
        self.tangent_norm_constant = tangent_norm_constant

    def forward(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        clean_images: torch.Tensor,
        text_embedding: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        global_step: int,
        model_kwargs: dict,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the CTCD loss.

        Args:
            student_model: Student denoiser (must support return_logvar and jvp)
            teacher_model: Teacher denoiser (frozen)
            clean_images: Clean latent images x_0 (already * sigma_data)
            text_embedding: Text conditioning
            text_mask: Text attention mask
            timesteps: Sampled timesteps in [0, pi/2] (trigonometric flow)
            global_step: Current training step for warmup
            model_kwargs: Additional kwargs for model forward
        """
        sd = self.sigma_data
        x0 = clean_images
        t = timesteps.view(-1, 1, 1, 1)
        z = torch.randn_like(x0) * sd

        # Noisy latent in trigonometric flow
        x_t = torch.cos(t) * x0 + torch.sin(t) * z

        # Teacher velocity field
        with torch.no_grad():
            teacher_pred = teacher_model(
                x_t / sd, t.flatten(),
                encoder_hidden_states=text_embedding,
                **model_kwargs,
            )
            if isinstance(teacher_pred, tuple):
                teacher_pred = teacher_pred[0]
            elif hasattr(teacher_pred, "sample"):
                teacher_pred = teacher_pred.sample
            dxt_dt = sd * teacher_pred

        # Tangent directions for JVP
        v_x = torch.cos(t) * torch.sin(t) * dxt_dt / sd
        v_t = torch.cos(t) * torch.sin(t)

        # Standard student forward (F_theta)
        student_out = student_model(
            x_t / sd, t.flatten(),
            encoder_hidden_states=text_embedding,
            **model_kwargs,
        )
        if isinstance(student_out, tuple):
            F_theta = student_out[0]
            logvar = student_out[1] if len(student_out) > 1 and student_out[1] is not None else None
        elif hasattr(student_out, "sample"):
            F_theta = student_out.sample
            logvar = None
        else:
            F_theta = student_out
            logvar = None

        # F_theta_minus (stop-gradient target)
        F_theta_minus = F_theta.detach()

        # Approximate JVP using finite differences when native JVP not available
        with torch.no_grad():
            eps_fd = 1e-3
            x_t_pert = x_t + eps_fd * v_x
            t_pert = t + eps_fd * v_t
            student_pert = student_model(
                x_t_pert / sd, t_pert.flatten(),
                encoder_hidden_states=text_embedding,
                **model_kwargs,
            )
            if isinstance(student_pert, tuple):
                F_pert = student_pert[0]
            elif hasattr(student_pert, "sample"):
                F_pert = student_pert.sample
            else:
                F_pert = student_pert
            F_theta_grad = (F_pert - F_theta_minus) / eps_fd

        # Warmup coefficient
        r = min(1.0, global_step / max(self.tangent_warmup_steps, 1))

        # Compute tangent g
        g = -torch.cos(t) * torch.cos(t) * (sd * F_theta_minus - dxt_dt)
        second_term = -r * (torch.cos(t) * torch.sin(t) * x_t + sd * F_theta_grad)
        g = g + second_term

        # Tangent normalization
        g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
        g = g / (g_norm + self.tangent_norm_constant)

        # Weighting
        sigma = torch.tan(t) * sd
        weight = 1.0 / sigma.clamp(min=1e-6)

        l2_loss = torch.square(F_theta - F_theta_minus - g)

        if logvar is not None:
            logvar = logvar.view(-1, 1, 1, 1)
            loss = (weight / torch.exp(logvar)) * l2_loss + logvar
        else:
            loss = weight * l2_loss

        loss = loss.mean()
        loss_no_logvar = (weight * l2_loss).mean()

        metrics = {
            "ctcd_loss": loss.item(),
            "ctcd_loss_no_logvar": loss_no_logvar.item(),
            "ctcd_g_norm": g_norm.mean().item(),
        }
        return loss, metrics


class LatentDiscriminator(nn.Module):
    """
    Latent-space discriminator for LADD (Latent Adversarial Diffusion Distillation).

    Operates on noised latents at various noise levels, using the teacher
    backbone as a feature extractor with trainable classification heads.

    This is a simplified version suitable for the project's scope — uses
    the teacher denoiser backbone (frozen) with a small trainable head.
    """

    def __init__(self, backbone: nn.Module, latent_channels: int, device: str,
                 deepcopy_fn=None):
        super().__init__()
        if deepcopy_fn is not None:
            self.backbone = deepcopy_fn(backbone)
        else:
            self.backbone = copy.deepcopy(backbone)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        # Trainable classification heads
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(latent_channels * 16, 512),
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        ).to(device)
        self.head.requires_grad_(True)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Get discriminator logits for the given latents."""
        with torch.no_grad():
            features = self.backbone(
                latents, timesteps,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )
            if isinstance(features, tuple):
                features = features[0]
            elif hasattr(features, "sample"):
                features = features.sample

        logits = self.head(features)
        return logits


class LADDLoss(nn.Module):
    """
    Latent Adversarial Diffusion Distillation (LADD) loss.

    The discriminator distinguishes between:
      - Real: noise-augmented ground-truth latents
      - Fake: noise-augmented student-predicted x_0

    Generator loss uses hinge or cross-entropy on fake samples.
    Discriminator loss uses standard real/fake classification.
    """

    def __init__(
        self,
        adv_lambda: float = 1.0,
        loss_type: str = "hinge",
        r1_penalty_weight: float = 0.0,
    ):
        super().__init__()
        self.adv_lambda = adv_lambda
        self.loss_type = loss_type
        self.r1_penalty_weight = r1_penalty_weight

    def generator_loss(
        self,
        pred_fake: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Adversarial loss for generator (fool discriminator)."""
        if self.loss_type == "hinge":
            loss = -torch.mean(pred_fake)
        else:  # cross_entropy
            loss = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))

        return loss * self.adv_lambda, {"adv_loss_G": loss.item()}

    def discriminator_loss(
        self,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Adversarial loss for discriminator."""
        if self.loss_type == "hinge":
            loss_real = torch.mean(F.relu(1.0 - pred_real))
            loss_fake = torch.mean(F.relu(1.0 + pred_fake))
            loss = 0.5 * (loss_real + loss_fake)
        else:  # cross_entropy
            loss_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
            loss_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
            loss = loss_real + loss_fake

        return loss, {
            "D_loss": loss.item(),
            "D_loss_real": loss_real.item(),
            "D_loss_fake": loss_fake.item(),
        }


# ============================================================================
#  Noise / Flow Utilities
# ============================================================================

def _get_x0_from_noise(
    noisy_latents: torch.Tensor,
    noise_pred: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Predict x_0 from noise prediction using DDPM formula."""
    alpha_t = alphas_cumprod[timesteps.long()].view(-1, 1, 1, 1)
    x0 = (noisy_latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt().clamp(min=1e-8)
    return x0


def _sample_trigflow_timesteps(
    batch_size: int,
    device: torch.device,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> torch.Tensor:
    """Sample timesteps for trigonometric flow (used by CTCD / SANA-Sprint)."""
    u = torch.randn(batch_size, device=device) * logit_std + logit_mean
    u = torch.sigmoid(u)  # uniform in [0, 1]
    t = u * (math.pi / 2)  # map to [0, pi/2]
    return t


# ============================================================================
#  Caption Dataset for Distillation Training
# ============================================================================

class CaptionNoiseDataset(Dataset):
    """
    Dataset that produces (noise, timestep, caption_embedding) tuples
    for distillation training. The noise and timesteps are generated on-the-fly.
    Caption embeddings are pre-computed to save memory.
    """

    def __init__(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor],
        latent_shape: Tuple[int, ...],
        num_train_timesteps: int = 1000,
    ):
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.latent_shape = latent_shape
        self.num_train_timesteps = num_train_timesteps

    def __len__(self):
        return self.prompt_embeds.shape[0]

    def __getitem__(self, idx):
        t = torch.randint(0, self.num_train_timesteps, (1,)).item()
        noise = torch.randn(self.latent_shape)
        embed = self.prompt_embeds[idx]
        pooled = self.pooled_prompt_embeds[idx] if self.pooled_prompt_embeds is not None else torch.tensor(0)
        return noise, t, embed, pooled


# ============================================================================
#  Utility Functions
# ============================================================================

def free_memory():
    """Free GPU memory aggressively."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_type(model_name: str) -> str:
    """Determine model type from model name."""
    name_lower = model_name.lower()
    if "flux" in name_lower:
        return "flux"
    elif "sdxl" in name_lower or "stable-diffusion-xl" in name_lower:
        return "sdxl"
    elif "sd3" in name_lower or "stable-diffusion-3" in name_lower:
        return "sd3"
    elif "sana" in name_lower:
        return "sana"
    else:
        return "sdxl"  # default


# ============================================================================
#  Knowledge Distillation Pipeline
# ============================================================================

class KnowledgeDistillationPipeline:
    """
    Knowledge Distillation pipeline for optimizing diffusion models.

    Implements two state-of-the-art approaches:

    DMD2 (kd_mode="dmd2"):
        - Distribution matching: real vs fake score gradient
        - GAN loss with classification head on fake-score UNet
        - Two-timescale update (dfake updated more frequently than generator)

    CTCD (kd_mode="ctcd"):
        - Continuous-time consistency distillation via JVP-based tangent
        - Trigonometric flow parameterization
        - Tangent normalization and warmup

    CTCD+LADD (kd_mode="ctcd_ladd"):
        - CTCD loss + latent adversarial distillation
        - Discriminator on noise-augmented latents
        - Alternating generator/discriminator updates

    Supports: SDXL, Flux, SD3, SANA model families.
    """

    def __init__(
        self,
        teacher_model_name: str = "SDXL",
        student_model_name: Optional[str] = None,
        student_num_blocks: Optional[int] = None,
        kd_mode: str = "dmd2",               # dmd2 | ctcd | ctcd_ladd
        # DMD2 specific
        dm_loss_weight: float = 1.0,
        gan_loss_weight: float = 1.0,
        real_guidance_scale: float = 6.0,
        dfake_gen_update_ratio: int = 5,
        # CTCD specific
        sigma_data: float = 1.0,
        tangent_warmup_steps: int = 1000,
        ctcd_logit_mean: float = 0.0,
        ctcd_logit_std: float = 1.0,
        # LADD specific
        adv_lambda: float = 0.1,
        scm_lambda: float = 1.0,
        ladd_loss_type: str = "hinge",
        # General training
        learning_rate: float = 1e-5,
        guidance_lr: Optional[float] = None,
        num_epochs: int = 5,
        batch_size: int = 1,
        num_train_timesteps: int = 1000,
        gradient_accumulation_steps: int = 4,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        memory_efficient: bool = False,
        device: str = "cuda",
    ):
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name or teacher_model_name
        self.student_num_blocks = student_num_blocks
        self.kd_mode = kd_mode

        # DMD2 params
        self.dm_loss_weight = dm_loss_weight
        self.gan_loss_weight = gan_loss_weight
        self.real_guidance_scale = real_guidance_scale
        self.dfake_gen_update_ratio = dfake_gen_update_ratio

        # CTCD params
        self.sigma_data = sigma_data
        self.tangent_warmup_steps = tangent_warmup_steps
        self.ctcd_logit_mean = ctcd_logit_mean
        self.ctcd_logit_std = ctcd_logit_std

        # LADD params
        self.adv_lambda = adv_lambda
        self.scm_lambda = scm_lambda
        self.ladd_loss_type = ladd_loss_type

        # General
        self.learning_rate = learning_rate
        self.guidance_lr = guidance_lr or learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_train_timesteps = num_train_timesteps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.memory_efficient = memory_efficient
        self.device = device

        # Will be filled during load
        self.teacher_pipeline = None
        self.student_pipeline = None
        self.teacher_denoiser = None
        self.student_denoiser = None
        self.scheduler = None
        self.ema_model = None
        self.model_type = None

        # DMD2 specific
        self.fake_score_net = None

        # LADD specific
        self.discriminator = None

        # Config from config.json
        self.config = self._load_config()

    def _load_config(self) -> dict:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
        )
        with open(config_path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load_teacher(self):
        """Load the teacher (full-size) diffusion pipeline."""
        print("\n" + "=" * 60)
        print("  Loading TEACHER model")
        print("=" * 60)

        _shared = _import_shared()
        _shared["setup_memory_optimizations"]()
        free_memory()

        models = self.config.get("models", {})
        model_info = models.get(self.teacher_model_name, {})
        model_path = model_info.get("path", self.teacher_model_name)
        self.model_type = get_model_type(model_path)

        print(f"  Model name : {self.teacher_model_name}")
        print(f"  Model path : {model_path}")
        print(f"  Model type : {self.model_type}")

        self.teacher_pipeline = self._load_pipeline(model_path, self.model_type)
        self.teacher_denoiser = self._get_denoiser(self.teacher_pipeline)
        self.scheduler = self.teacher_pipeline.scheduler

        # Freeze teacher
        self.teacher_denoiser.eval()
        for p in self.teacher_denoiser.parameters():
            p.requires_grad = False

        teacher_size = get_model_size_mb(self.teacher_denoiser)
        teacher_params = count_parameters(self.teacher_denoiser)
        print(f"  Teacher size   : {teacher_size:.1f} MB")
        print(f"  Teacher params : {teacher_params:,}")

    # ------------------------------------------------------------------
    # Memory-efficient offloading helpers
    # ------------------------------------------------------------------

    def _offload_to_cpu(self, model: nn.Module, name: str = ""):
        """Move model to CPU and free GPU cache."""
        if model is not None:
            model.to("cpu")
            torch.cuda.empty_cache()
            if name:
                print(f"    [mem] {name} -> CPU")

    def _load_to_gpu(self, model: nn.Module, name: str = ""):
        """Move model to GPU."""
        if model is not None:
            model.to(self.device)
            if name:
                print(f"    [mem] {name} -> GPU")

    def _load_fresh_denoiser(self) -> nn.Module:
        """Load a fresh copy of the denoiser by re-loading from pretrained weights.
        Used when copy.deepcopy fails (e.g. quantized nunchaku models)."""
        models = self.config.get("models", {})
        model_info = models.get(self.teacher_model_name, {})
        model_path = model_info.get("path", self.teacher_model_name)
        print(f"  Loading fresh denoiser from {model_path}...")
        pipeline = self._load_pipeline(model_path, self.model_type)
        denoiser = self._get_denoiser(pipeline)
        # Release the rest of the pipeline to save memory
        del pipeline
        free_memory()
        return denoiser

    def _safe_deepcopy_denoiser(self, model: nn.Module) -> nn.Module:
        """Try deepcopy first; if it fails (e.g. quantized model), load a fresh copy."""
        try:
            return copy.deepcopy(model)
        except (TypeError, RuntimeError) as e:
            print(f"  deepcopy failed ({e}), loading fresh model instead...")
            return self._load_fresh_denoiser()

    def create_student(self):
        """
        Create the student model. Strategy depends on configuration:
        - If student_num_blocks is set: create a slimmer version by removing blocks
        - Otherwise: deep-copy the teacher as starting point
        """
        print("\n" + "=" * 60)
        print("  Creating STUDENT model")
        print("=" * 60)

        if self.student_num_blocks is not None:
            self.student_denoiser = self._create_slim_student(self.student_num_blocks)
        else:
            # Start from a copy of the teacher
            self.student_denoiser = self._safe_deepcopy_denoiser(self.teacher_denoiser)
            self.student_denoiser.train()
            for p in self.student_denoiser.parameters():
                p.requires_grad = True

        # Create EMA copy
        if self.use_ema:
            self.ema_model = self._safe_deepcopy_denoiser(self.student_denoiser)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad = False

        student_size = get_model_size_mb(self.student_denoiser)
        student_params = count_parameters(self.student_denoiser)
        print(f"  Student size   : {student_size:.1f} MB")
        print(f"  Student params : {student_params:,}")
        print(f"  EMA enabled    : {self.use_ema}")
        print(f"  Memory efficient: {self.memory_efficient}")

        # Memory-efficient: offload teacher to CPU before creating more models
        if self.memory_efficient:
            self._offload_to_cpu(self.teacher_denoiser, "teacher")
            free_memory()

        # Create mode-specific auxiliary networks
        if self.kd_mode == "dmd2":
            print("  Creating Fake Score Network (DMD2)...")
            self.fake_score_net = FakeScoreNetwork(
                self.teacher_denoiser, self.model_type, self.device,
                deepcopy_fn=self._safe_deepcopy_denoiser,
            )
            if not self.memory_efficient:
                self.fake_score_net = self.fake_score_net.to(self.device)
            else:
                # Keep on CPU, will be moved to GPU when needed
                self.fake_score_net = self.fake_score_net.to("cpu")
                print("    [mem] fake_score_net -> CPU")

        if self.kd_mode == "ctcd_ladd":
            print("  Creating Latent Discriminator (LADD)...")
            latent_ch = 4 if self.model_type == "sdxl" else 16
            self.discriminator = LatentDiscriminator(
                self.teacher_denoiser, latent_ch, self.device,
                deepcopy_fn=self._safe_deepcopy_denoiser,
            )
            if not self.memory_efficient:
                self.discriminator = self.discriminator.to(self.device)
            else:
                self.discriminator = self.discriminator.to("cpu")
                print("    [mem] discriminator -> CPU")

        # Memory-efficient: EMA always on CPU
        if self.memory_efficient and self.use_ema and self.ema_model is not None:
            self._offload_to_cpu(self.ema_model, "ema_model")

        if self.memory_efficient:
            free_memory()
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            print(f"  [mem] GPU VRAM used after setup: {vram_used:.1f} GB")

    def _create_slim_student(self, num_blocks: int) -> nn.Module:
        """
        Create a smaller student by removing transformer/UNet blocks.
        This works by selecting every N-th block to keep.
        """
        student = self._safe_deepcopy_denoiser(self.teacher_denoiser)

        if self.model_type == "sdxl":
            if hasattr(student, "down_blocks"):
                original_count = len(student.down_blocks)
                if num_blocks < original_count:
                    keep_indices = torch.linspace(0, original_count - 1, num_blocks).long().tolist()
                    student.down_blocks = nn.ModuleList(
                        [student.down_blocks[i] for i in keep_indices]
                    )
                    print(f"  Trimmed down_blocks: {original_count} -> {num_blocks}")
            if hasattr(student, "up_blocks"):
                original_count = len(student.up_blocks)
                if num_blocks < original_count:
                    keep_indices = torch.linspace(0, original_count - 1, num_blocks).long().tolist()
                    student.up_blocks = nn.ModuleList(
                        [student.up_blocks[i] for i in keep_indices]
                    )
                    print(f"  Trimmed up_blocks: {original_count} -> {num_blocks}")

        elif self.model_type in ("flux", "sd3"):
            if hasattr(student, "transformer_blocks"):
                original_count = len(student.transformer_blocks)
                if num_blocks < original_count:
                    keep_indices = torch.linspace(0, original_count - 1, num_blocks).long().tolist()
                    student.transformer_blocks = nn.ModuleList(
                        [student.transformer_blocks[i] for i in keep_indices]
                    )
                    print(f"  Trimmed transformer_blocks: {original_count} -> {num_blocks}")

        student.train()
        for p in student.parameters():
            p.requires_grad = True
        return student

    def _load_pipeline(self, model_path: str, model_type: str):
        """Load the appropriate diffusers pipeline for the model type.
        
        NOTE: Distillation requires full-precision models (not quantized int4)
        because the denoiser must be deepcopy-able for student/EMA/fake-score
        networks, and quantized nunchaku models cannot be pickled.

        Loading strategy per model type (referenced from quantization/quantized.py):
        - SDXL: float16 + variant="fp16" + use_safetensors
        - Flux: bfloat16 + use_safetensors
        - SD3:  float16 + use_safetensors
        - SANA: bfloat16 + use_safetensors
        """
        from diffusers import (
            StableDiffusionXLPipeline,
            FluxPipeline,
            StableDiffusion3Pipeline,
        )

        if model_type == "sdxl":
            print(f"  Loading SDXL pipeline (float16)...")
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

        elif model_type == "flux":
            print(f"  Loading Flux pipeline (bfloat16)...")
            pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            )

        elif model_type == "sd3":
            print(f"  Loading SD3 pipeline (float16)...")
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

        elif model_type == "sana":
            try:
                from diffusers import SanaPipeline
                print(f"  Loading SANA pipeline (bfloat16)...")
                pipeline = SanaPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                )
            except ImportError:
                print("  SanaPipeline not available, falling back to SDXL loader")
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                )
        else:
            print(f"  Loading default SDXL pipeline (float16)...")
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

        pipeline.enable_model_cpu_offload()
        return pipeline

    @staticmethod
    def _get_denoiser(pipeline) -> nn.Module:
        """Extract the denoising backbone (UNet or Transformer) from a pipeline."""
        if hasattr(pipeline, "unet"):
            return pipeline.unet
        elif hasattr(pipeline, "transformer"):
            return pipeline.transformer
        else:
            raise ValueError("Pipeline has neither 'unet' nor 'transformer' attribute.")

    # ------------------------------------------------------------------
    # Text Embedding Pre-computation
    # ------------------------------------------------------------------

    def encode_prompts(self, captions: List[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Pre-compute text embeddings for all captions using the teacher pipeline's
        text encoders. Returns (prompt_embeds, pooled_prompt_embeds).
        """
        print(f"\n  Encoding {len(captions)} prompts...")
        all_embeds = []
        all_pooled = []

        with torch.no_grad():
            for i in range(0, len(captions), self.batch_size):
                batch = captions[i : i + self.batch_size]

                if self.model_type == "sdxl":
                    result = self.teacher_pipeline.encode_prompt(
                        prompt=batch,
                        device=self.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    if isinstance(result, tuple) and len(result) >= 2:
                        embeds, _, pooled, _ = result
                    else:
                        embeds = result
                        pooled = None
                elif self.model_type == "flux":
                    result = self.teacher_pipeline.encode_prompt(
                        prompt=batch,
                        prompt_2=batch,
                    )
                    if isinstance(result, tuple):
                        embeds = result[0]
                        pooled = result[1] if len(result) > 1 else None
                    else:
                        embeds = result
                        pooled = None
                else:
                    result = self.teacher_pipeline.encode_prompt(
                        prompt=batch,
                        device=self.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    if isinstance(result, tuple):
                        embeds = result[0]
                        pooled = result[1] if len(result) > 1 else None
                    else:
                        embeds = result
                        pooled = None

                all_embeds.append(embeds.cpu())
                if pooled is not None:
                    all_pooled.append(pooled.cpu())

                if (i // self.batch_size) % 50 == 0:
                    print(f"    Encoded {min(i + self.batch_size, len(captions))}/{len(captions)}")

        prompt_embeds = torch.cat(all_embeds, dim=0)
        pooled_prompt_embeds = torch.cat(all_pooled, dim=0) if all_pooled else None
        print(f"  Prompt embeds shape: {prompt_embeds.shape}")
        return prompt_embeds, pooled_prompt_embeds

    # ------------------------------------------------------------------
    # Training Loops
    # ------------------------------------------------------------------

    def train(self, captions: List[str], output_dir: str):
        """
        Main distillation training loop. Dispatches to mode-specific training.
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.kd_mode == "dmd2":
            return self._train_dmd2(captions, output_dir)
        elif self.kd_mode == "ctcd":
            return self._train_ctcd(captions, output_dir, use_ladd=False)
        elif self.kd_mode == "ctcd_ladd":
            return self._train_ctcd(captions, output_dir, use_ladd=True)
        else:
            raise ValueError(f"Unknown kd_mode: {self.kd_mode}. Choose from: dmd2, ctcd, ctcd_ladd")

    def _train_dmd2(self, captions: List[str], output_dir: str):
        """
        DMD2 training loop.

        Two-timescale update:
        1. Every step: update fake score network (denoising + GAN discriminator loss)
        2. Every dfake_gen_update_ratio steps: update generator (DM + GAN loss)
        """
        log_path = os.path.join(output_dir, "training_log.json")

        # 1. Encode prompts
        prompt_embeds, pooled_prompt_embeds = self.encode_prompts(captions)

        # 2. Latent shape
        latent_shape = self._get_latent_shape()

        # 3. Dataset
        dataset = CaptionNoiseDataset(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            latent_shape=latent_shape,
            num_train_timesteps=self.num_train_timesteps,
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
        )

        # 4. Loss
        dmd2_loss = DMD2Loss(
            dm_loss_weight=self.dm_loss_weight,
            gan_loss_weight=self.gan_loss_weight,
            real_guidance_scale=self.real_guidance_scale,
        )

        # 5. Optimizers (two-timescale)
        optimizer_G = torch.optim.AdamW(
            self.student_denoiser.parameters(),
            lr=self.learning_rate, weight_decay=1e-2,
        )
        guidance_params = list(self.fake_score_net.parameters())
        optimizer_D = torch.optim.AdamW(
            guidance_params, lr=self.guidance_lr, weight_decay=1e-2,
        )

        total_steps = len(dataloader) * self.num_epochs // self.gradient_accumulation_steps
        lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_G, T_max=max(total_steps, 1), eta_min=self.learning_rate * 0.01,
        )

        # Get alphas_cumprod
        if hasattr(self.scheduler, "alphas_cumprod"):
            alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        else:
            alphas_cumprod = torch.linspace(1.0, 0.01, self.num_train_timesteps).to(self.device)

        # 6. Training
        print("\n" + "=" * 60)
        print("  Starting DMD2 Distillation Training")
        print("=" * 60)
        print(f"  Mode               : {self.kd_mode}")
        print(f"  Epochs             : {self.num_epochs}")
        print(f"  Batch size         : {self.batch_size}")
        print(f"  Generator LR       : {self.learning_rate}")
        print(f"  Guidance LR        : {self.guidance_lr}")
        print(f"  DM loss weight     : {self.dm_loss_weight}")
        print(f"  GAN loss weight    : {self.gan_loss_weight}")
        print(f"  D/G update ratio   : {self.dfake_gen_update_ratio}")
        print(f"  Latent shape       : {latent_shape}")
        print()

        training_log = []
        global_step = 0
        best_loss = float("inf")

        self.teacher_denoiser.eval()
        self.student_denoiser.to(self.device)

        # Memory-efficient: enable gradient checkpointing on student
        if self.memory_efficient:
            if hasattr(self.student_denoiser, 'enable_gradient_checkpointing'):
                self.student_denoiser.enable_gradient_checkpointing()
                print("  [mem] Gradient checkpointing enabled on student")

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_metrics = {}
            self.student_denoiser.train()
            if not self.memory_efficient:
                self.fake_score_net.train()

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for step, (noise, timesteps_raw, embeds, pooled) in enumerate(pbar):
                noise = noise.to(self.device, dtype=torch.float16)
                embeds = embeds.to(self.device, dtype=torch.float16)
                if pooled.dim() > 1:
                    pooled = pooled.to(self.device, dtype=torch.float16)
                else:
                    pooled = None

                forward_kwargs = self._get_forward_kwargs(pooled, noise.shape[0])
                is_generator_step = (global_step % self.dfake_gen_update_ratio == 0)

                # Step 1: Generate images with student (feedforward)
                conditioning_timestep = torch.ones(
                    noise.shape[0], device=self.device, dtype=torch.long
                ) * (self.num_train_timesteps - 1)

                if is_generator_step:
                    gen_noise_pred = self._forward_denoiser(
                        self.student_denoiser, noise, conditioning_timestep, embeds, pooled,
                    )
                else:
                    with torch.no_grad():
                        gen_noise_pred = self._forward_denoiser(
                            self.student_denoiser, noise, conditioning_timestep, embeds, pooled,
                        )

                generated_latents = _get_x0_from_noise(
                    noise.double(), gen_noise_pred.double(),
                    alphas_cumprod.double(), conditioning_timestep,
                ).float()

                # Step 2: Generator update (DM + GAN loss)
                if is_generator_step:
                    # Memory-efficient: bring teacher + fake_score to GPU for DM loss
                    if self.memory_efficient:
                        self._load_to_gpu(self.teacher_denoiser)
                        self._load_to_gpu(self.fake_score_net)

                    # Distribution matching loss
                    loss_dm, dm_metrics = dmd2_loss.compute_distribution_matching_loss(
                        generated_latents, self.teacher_denoiser,
                        self.fake_score_net, embeds,
                        self.scheduler, alphas_cumprod,
                        self.num_train_timesteps, self.model_type,
                        self.device, forward_kwargs,
                    )

                    # Memory-efficient: teacher no longer needed, offload
                    if self.memory_efficient:
                        self._offload_to_cpu(self.teacher_denoiser)

                    # GAN loss for generator (only needs fake_score_net, already on GPU)
                    loss_gan_g, gan_g_metrics = dmd2_loss.compute_gan_loss_generator(
                        generated_latents, self.fake_score_net,
                    )

                    # Memory-efficient: offload fake_score_net after generator step
                    if self.memory_efficient:
                        self._offload_to_cpu(self.fake_score_net)

                    gen_loss = (loss_dm + loss_gan_g) / self.gradient_accumulation_steps
                    gen_loss.backward()

                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.student_denoiser.parameters(), 10.0)
                        optimizer_G.step()
                        lr_scheduler_G.step()
                        optimizer_G.zero_grad()
                        optimizer_D.zero_grad()

                        if self.use_ema and self.ema_model is not None:
                            self._update_ema()

                    metrics = {**dm_metrics, **gan_g_metrics}
                    epoch_loss += (loss_dm.item() + loss_gan_g.item())
                else:
                    metrics = {}

                # Step 3: Guidance/discriminator update (every step)
                # Memory-efficient: bring fake_score_net to GPU for its update
                if self.memory_efficient:
                    self._load_to_gpu(self.fake_score_net)
                    self.fake_score_net.train()

                # Fake score denoising loss
                loss_fake, fake_metrics = dmd2_loss.compute_fake_score_loss(
                    generated_latents, self.fake_score_net,
                    self.scheduler, embeds,
                    self.num_train_timesteps, self.device, forward_kwargs,
                )

                guidance_loss = loss_fake / self.gradient_accumulation_steps
                guidance_loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.fake_score_net.parameters(), 10.0)
                    optimizer_D.step()
                    optimizer_D.zero_grad()
                    optimizer_G.zero_grad()

                # Memory-efficient: offload fake_score_net after its update
                if self.memory_efficient:
                    self._offload_to_cpu(self.fake_score_net)

                metrics.update(fake_metrics)
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v

                pbar.set_postfix(
                    dm=f"{metrics.get('loss_dm', 0):.4f}",
                    fake=f"{metrics.get('loss_fake_score', 0):.4f}",
                )

                global_step += 1
                if step % 50 == 0:
                    free_memory()

            # End of epoch
            avg_loss = epoch_loss / max(len(dataloader), 1)
            avg_metrics = {k: v / max(len(dataloader), 1) for k, v in epoch_metrics.items()}

            log_entry = {"epoch": epoch + 1, "avg_loss": avg_loss, "lr": optimizer_G.param_groups[0]["lr"], **avg_metrics}
            training_log.append(log_entry)

            print(f"\n  Epoch {epoch + 1} Summary:")
            print(f"    Avg Loss : {avg_loss:.6f}")
            for k, v in avg_metrics.items():
                print(f"    {k}: {v:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint(output_dir, epoch + 1, is_best=True)

            if (epoch + 1) % max(1, self.num_epochs // 3) == 0:
                self._save_checkpoint(output_dir, epoch + 1)

        self._save_checkpoint(output_dir, self.num_epochs, is_final=True)
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        print(f"\n  DMD2 Training complete! Logs saved to {log_path}")
        return training_log

    def _train_ctcd(self, captions: List[str], output_dir: str, use_ladd: bool = False):
        """
        CTCD / CTCD+LADD training loop (SANA-Sprint).

        If use_ladd=True: alternating G/D phases with combined CTCD + adversarial loss.
        """
        log_path = os.path.join(output_dir, "training_log.json")

        # 1. Encode prompts
        prompt_embeds, pooled_prompt_embeds = self.encode_prompts(captions)

        # 2. Latent shape & dataset
        latent_shape = self._get_latent_shape()
        dataset = CaptionNoiseDataset(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            latent_shape=latent_shape,
            num_train_timesteps=self.num_train_timesteps,
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
        )

        # 3. Loss functions
        ctcd_loss_fn = CTCDLoss(
            sigma_data=self.sigma_data,
            tangent_warmup_steps=self.tangent_warmup_steps,
        )
        ladd_loss_fn = LADDLoss(
            adv_lambda=self.adv_lambda,
            loss_type=self.ladd_loss_type,
        ) if use_ladd else None

        # 4. Optimizers
        optimizer_G = torch.optim.AdamW(
            self.student_denoiser.parameters(),
            lr=self.learning_rate, weight_decay=1e-2,
        )

        optimizer_D = None
        if use_ladd and self.discriminator is not None:
            optimizer_D = torch.optim.AdamW(
                self.discriminator.head.parameters(),
                lr=self.guidance_lr, weight_decay=1e-2,
            )

        total_steps = len(dataloader) * self.num_epochs // self.gradient_accumulation_steps
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_G, T_max=max(total_steps, 1), eta_min=self.learning_rate * 0.01,
        )

        # 5. Print config
        mode_name = "CTCD + LADD" if use_ladd else "CTCD"
        print("\n" + "=" * 60)
        print(f"  Starting {mode_name} Distillation Training")
        print("=" * 60)
        print(f"  Mode             : {self.kd_mode}")
        print(f"  Epochs           : {self.num_epochs}")
        print(f"  Batch size       : {self.batch_size}")
        print(f"  Learning rate    : {self.learning_rate}")
        print(f"  Sigma data       : {self.sigma_data}")
        print(f"  Tangent warmup   : {self.tangent_warmup_steps}")
        if use_ladd:
            print(f"  LADD lambda      : {self.adv_lambda}")
            print(f"  SCM lambda       : {self.scm_lambda}")
            print(f"  LADD loss type   : {self.ladd_loss_type}")
        print(f"  Latent shape     : {latent_shape}")
        print()

        training_log = []
        global_step = 0
        best_loss = float("inf")
        phase = "G"

        self.teacher_denoiser.eval()
        self.student_denoiser.to(self.device)

        # Memory-efficient: enable gradient checkpointing on student
        if self.memory_efficient:
            if hasattr(self.student_denoiser, 'enable_gradient_checkpointing'):
                self.student_denoiser.enable_gradient_checkpointing()
                print("  [mem] Gradient checkpointing enabled on student")

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_metrics = {}
            self.student_denoiser.train()

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for step, (noise, _, embeds, pooled) in enumerate(pbar):
                noise = noise.to(self.device, dtype=torch.float16)
                embeds = embeds.to(self.device, dtype=torch.float16)
                if pooled.dim() > 1:
                    pooled = pooled.to(self.device, dtype=torch.float16)
                else:
                    pooled = None

                sd = self.sigma_data
                clean_images = torch.randn_like(noise) * sd  # Random clean in latent space
                model_kwargs = self._get_forward_kwargs(pooled, noise.shape[0])

                # Sample timesteps (trigonometric flow)
                timesteps = _sample_trigflow_timesteps(
                    noise.shape[0], self.device,
                    logit_mean=self.ctcd_logit_mean,
                    logit_std=self.ctcd_logit_std,
                )

                if phase == "G":
                    if use_ladd and self.discriminator is not None:
                        if not self.memory_efficient:
                            self.discriminator.eval()
                    self.student_denoiser.train()

                    # Memory-efficient: bring teacher to GPU for CTCD loss
                    if self.memory_efficient:
                        self._load_to_gpu(self.teacher_denoiser)

                    # CTCD loss
                    ctcd_l, ctcd_m = ctcd_loss_fn(
                        self.student_denoiser, self.teacher_denoiser,
                        clean_images, embeds, None, timesteps,
                        global_step, model_kwargs,
                    )

                    # Memory-efficient: offload teacher after CTCD
                    if self.memory_efficient:
                        self._offload_to_cpu(self.teacher_denoiser)

                    total_loss = self.scm_lambda * ctcd_l

                    # LADD adversarial loss for generator
                    if use_ladd and self.discriminator is not None:
                        # Memory-efficient: bring discriminator to GPU
                        if self.memory_efficient:
                            self._load_to_gpu(self.discriminator)
                            self.discriminator.eval()

                        t_view = timesteps.view(-1, 1, 1, 1)
                        x_t = torch.cos(t_view) * clean_images + torch.sin(t_view) * noise * sd

                        with torch.no_grad():
                            student_pred = self._forward_denoiser(
                                self.student_denoiser, x_t / sd, timesteps, embeds, pooled,
                            )
                        pred_x0 = torch.cos(t_view) * x_t - torch.sin(t_view) * student_pred * sd

                        # Noise augment pred_x0 for discriminator
                        t_D = _sample_trigflow_timesteps(noise.shape[0], self.device, 0.0, 1.0)
                        t_D_view = t_D.view(-1, 1, 1, 1)
                        z_D = torch.randn_like(pred_x0) * sd
                        noised_pred = torch.cos(t_D_view) * pred_x0 + torch.sin(t_D_view) * z_D

                        pred_fake = self.discriminator(
                            noised_pred / sd, t_D,
                            encoder_hidden_states=embeds,
                            **model_kwargs,
                        )
                        adv_l, adv_m = ladd_loss_fn.generator_loss(pred_fake)
                        total_loss = total_loss + adv_l
                        ctcd_m.update(adv_m)

                        # Memory-efficient: offload discriminator
                        if self.memory_efficient:
                            self._offload_to_cpu(self.discriminator)

                    total_loss = total_loss / self.gradient_accumulation_steps
                    total_loss.backward()

                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.student_denoiser.parameters(), 10.0)
                        optimizer_G.step()
                        lr_scheduler.step()
                        optimizer_G.zero_grad()
                        if optimizer_D is not None:
                            optimizer_D.zero_grad()

                        if self.use_ema and self.ema_model is not None:
                            self._update_ema()

                        if use_ladd:
                            phase = "D"

                        global_step += 1

                    metrics = ctcd_m
                    epoch_loss += total_loss.item() * self.gradient_accumulation_steps

                elif phase == "D" and use_ladd and self.discriminator is not None:
                    # Memory-efficient: bring discriminator to GPU
                    if self.memory_efficient:
                        self._load_to_gpu(self.discriminator)

                    self.discriminator.train()
                    self.student_denoiser.eval()

                    t_view = timesteps.view(-1, 1, 1, 1)
                    x_t = torch.cos(t_view) * clean_images + torch.sin(t_view) * noise * sd

                    with torch.no_grad():
                        student_pred = self._forward_denoiser(
                            self.student_denoiser, x_t / sd, timesteps, embeds, pooled,
                        )
                        pred_x0 = torch.cos(t_view) * x_t - torch.sin(t_view) * student_pred * sd

                    # Noise augment for discriminator
                    t_D_fake = _sample_trigflow_timesteps(noise.shape[0], self.device, 0.0, 1.0)
                    t_D_real = _sample_trigflow_timesteps(noise.shape[0], self.device, 0.0, 1.0)
                    z_D_fake = torch.randn_like(pred_x0) * sd
                    z_D_real = torch.randn_like(clean_images) * sd

                    noised_fake = torch.cos(t_D_fake.view(-1,1,1,1)) * pred_x0 + torch.sin(t_D_fake.view(-1,1,1,1)) * z_D_fake
                    noised_real = torch.cos(t_D_real.view(-1,1,1,1)) * clean_images + torch.sin(t_D_real.view(-1,1,1,1)) * z_D_real

                    pred_fake = self.discriminator(noised_fake / sd, t_D_fake, encoder_hidden_states=embeds, **model_kwargs)
                    pred_real = self.discriminator(noised_real / sd, t_D_real, encoder_hidden_states=embeds, **model_kwargs)

                    d_loss, d_metrics = ladd_loss_fn.discriminator_loss(pred_real, pred_fake)
                    d_loss = d_loss / self.gradient_accumulation_steps
                    d_loss.backward()

                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.discriminator.head.parameters(), 10.0)
                        optimizer_D.step()
                        optimizer_D.zero_grad()
                        optimizer_G.zero_grad()
                        phase = "G"
                        global_step += 1

                    # Memory-efficient: offload discriminator
                    if self.memory_efficient:
                        self._offload_to_cpu(self.discriminator)

                    metrics = d_metrics
                else:
                    metrics = {}
                    global_step += 1

                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v

                pbar.set_postfix(
                    phase=phase,
                    loss=f"{metrics.get('ctcd_loss', metrics.get('D_loss', 0)):.4f}",
                )

                if step % 50 == 0:
                    free_memory()

            # End of epoch
            avg_loss = epoch_loss / max(len(dataloader), 1)
            avg_metrics = {k: v / max(len(dataloader), 1) for k, v in epoch_metrics.items()}

            log_entry = {"epoch": epoch + 1, "avg_loss": avg_loss, "lr": optimizer_G.param_groups[0]["lr"], **avg_metrics}
            training_log.append(log_entry)

            print(f"\n  Epoch {epoch + 1} Summary:")
            print(f"    Avg Loss : {avg_loss:.6f}")
            for k, v in avg_metrics.items():
                print(f"    {k}: {v:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint(output_dir, epoch + 1, is_best=True)

            if (epoch + 1) % max(1, self.num_epochs // 3) == 0:
                self._save_checkpoint(output_dir, epoch + 1)

        self._save_checkpoint(output_dir, self.num_epochs, is_final=True)
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        print(f"\n  {mode_name} Training complete! Logs saved to {log_path}")
        return training_log

    # ------------------------------------------------------------------
    # Helpers for training
    # ------------------------------------------------------------------

    def _get_latent_shape(self) -> Tuple[int, ...]:
        if self.model_type == "sdxl":
            return (4, 128, 128)
        elif self.model_type == "flux":
            return (16, 64, 64)
        elif self.model_type == "sd3":
            return (16, 128, 128)
        else:
            return (4, 128, 128)

    def _get_forward_kwargs(self, pooled: Optional[torch.Tensor], batch_size: int) -> dict:
        """Build model-specific forward kwargs."""
        kwargs = {}
        if self.model_type == "sdxl" and pooled is not None:
            time_ids = torch.zeros(batch_size, 6, device=self.device, dtype=pooled.dtype)
            kwargs["added_cond_kwargs"] = {
                "text_embeds": pooled,
                "time_ids": time_ids,
            }
        return kwargs

    def _forward_denoiser(
        self,
        model: nn.Module,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the denoising model, handling different architectures."""
        kwargs = self._get_forward_kwargs(pooled, latent.shape[0])

        output = model(latent, timesteps, encoder_hidden_states=encoder_hidden_states, **kwargs)

        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output

    def _update_ema(self):
        """Update EMA model parameters. Works even when EMA is on CPU."""
        for ema_p, student_p in zip(self.ema_model.parameters(), self.student_denoiser.parameters()):
            # Move student param to same device as EMA for the update
            ema_p.data.mul_(self.ema_decay).add_(student_p.data.to(ema_p.device), alpha=1 - self.ema_decay)

    def _save_checkpoint(self, output_dir: str, epoch: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        model_to_save = self.ema_model if (self.use_ema and self.ema_model is not None) else self.student_denoiser

        state = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "kd_mode": self.kd_mode,
            "teacher_model": self.teacher_model_name,
            "student_model": self.student_model_name,
        }

        if is_best:
            path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(state, path)
            print(f"    Saved best checkpoint: {path}")
        if is_final:
            path = os.path.join(ckpt_dir, "final_model.pt")
            torch.save(state, path)
            print(f"    Saved final checkpoint: {path}")
        else:
            path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(state, path)
            print(f"    Saved checkpoint: {path}")

    # ------------------------------------------------------------------
    # Inference / Evaluation
    # ------------------------------------------------------------------

    def load_student_checkpoint(self, checkpoint_path: str):
        """Load a trained student checkpoint."""
        print(f"\n  Loading student checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.student_denoiser.load_state_dict(state["model_state_dict"])
        self.student_denoiser.eval()
        print(f"  Loaded checkpoint from epoch {state.get('epoch', '?')}")

    def replace_pipeline_denoiser(self):
        """Replace the teacher's denoiser with the trained student in the pipeline."""
        if hasattr(self.teacher_pipeline, "unet"):
            self.teacher_pipeline.unet = self.student_denoiser
        elif hasattr(self.teacher_pipeline, "transformer"):
            self.teacher_pipeline.transformer = self.student_denoiser
        self.student_pipeline = self.teacher_pipeline
        print("  Pipeline denoiser replaced with student model.")

    def generate_images(
        self,
        captions_dict: Dict[str, str],
        output_dir: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ):
        """
        Generate images using the student model pipeline.
        Follows the same pattern as other modules (quantization, pruning, etc.).
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.student_pipeline is None:
            self.replace_pipeline_denoiser()

        pipeline = self.student_pipeline

        print(f"\n=== Generating {len(captions_dict)} images with distilled student ===\n")
        _shared = _import_shared()
        generation_times = {}

        for i, (filename, prompt) in enumerate(tqdm(captions_dict.items(), desc="Generating")):
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                print(f"  Skipping {filename} (already exists)")
                continue

            try:
                gen_time, metadata = _shared["generate_image_and_monitor"](
                    pipeline, prompt, output_path, filename,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
                generation_times[filename] = gen_time
            except Exception as e:
                print(f"  Error generating {filename}: {e}")
                generation_times[filename] = -1

            if i % 10 == 0:
                free_memory()

        return generation_times

    def evaluate(
        self,
        generated_dir: str,
        original_dir: str,
        captions_dict: Dict[str, str],
        image_dimensions: Optional[Dict] = None,
        metrics_subset: int = 100,
    ) -> Dict[str, Any]:
        """
        Evaluate the student model's output quality using standard metrics:
        FID, CLIP Score, ImageReward, LPIPS, PSNR.
        """
        print("\n" + "=" * 60)
        print("  Evaluating Distilled Student Model")
        print("=" * 60)

        _shared = _import_shared()
        results = {}

        # Resize images for metrics
        resized_dir = os.path.join(generated_dir, "resized")
        os.makedirs(resized_dir, exist_ok=True)
        if image_dimensions:
            try:
                _shared["resize_images"](generated_dir, resized_dir, image_dimensions)
            except Exception as e:
                print(f"  Error resizing: {e}")

        # FID
        try:
            results["fid"] = _shared["calculate_fid_subset"](generated_dir, resized_dir, original_dir)
            print(f"  FID Score: {results['fid']}")
        except Exception as e:
            print(f"  FID error: {e}")

        # CLIP Score
        try:
            generated_files = [
                f for f in os.listdir(generated_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            available_captions = {
                fn: cap for fn, cap in captions_dict.items() if fn in generated_files
            }
            results["clip_score"] = _shared["calculate_clip_score"](generated_dir, available_captions)
            print(f"  CLIP Score: {results['clip_score']}")
        except Exception as e:
            print(f"  CLIP error: {e}")

        # ImageReward
        try:
            results["image_reward"] = _shared["compute_image_reward"](generated_dir, captions_dict)
            print(f"  ImageReward: {results['image_reward']}")
        except Exception as e:
            print(f"  ImageReward error: {e}")

        # LPIPS
        try:
            results["lpips"] = _shared["calculate_lpips"](original_dir, generated_dir, metrics_subset)
            print(f"  LPIPS: {results['lpips']}")
        except Exception as e:
            print(f"  LPIPS error: {e}")

        # PSNR
        try:
            filenames = list(captions_dict.keys())[:metrics_subset]
            results["psnr"] = _shared["calculate_psnr_resized"](original_dir, resized_dir, filenames)
            print(f"  PSNR: {results['psnr']}")
        except Exception as e:
            print(f"  PSNR error: {e}")

        # Save results
        results_path = os.path.join(generated_dir, "distillation_metrics.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Metrics saved to {results_path}")

        return results

    def get_compression_stats(self) -> Dict[str, Any]:
        """Report compression statistics between teacher and student."""
        stats = {}
        if self.teacher_denoiser is not None:
            stats["teacher_size_mb"] = get_model_size_mb(self.teacher_denoiser)
            stats["teacher_params"] = count_parameters(self.teacher_denoiser)
        if self.student_denoiser is not None:
            stats["student_size_mb"] = get_model_size_mb(self.student_denoiser)
            stats["student_params"] = count_parameters(self.student_denoiser)
        if "teacher_size_mb" in stats and "student_size_mb" in stats:
            stats["size_reduction_pct"] = (
                100 * (stats["teacher_size_mb"] - stats["student_size_mb"]) / stats["teacher_size_mb"]
            )
            stats["param_reduction_pct"] = (
                100 * (stats["teacher_params"] - stats["student_params"]) / stats["teacher_params"]
            )
        return stats


# ============================================================================
#  Main entry point
# ============================================================================

def main(args=None):
    """
    Main function for Knowledge Distillation module.
    Can be called standalone or from app.py.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Knowledge Distillation for Diffusion Models (DMD2 / SANA-Sprint)")

        # Model configuration
        parser.add_argument("--teacher_model", type=str, default="SDXL",
                            help="Teacher model name from config.json (e.g., SDXL, Flux.1-schnell, SD3)")
        parser.add_argument("--student_model", type=str, default=None,
                            help="Student model name (defaults to teacher)")
        parser.add_argument("--student_num_blocks", type=int, default=None,
                            help="Number of blocks for slim student (None = copy teacher)")

        # KD mode
        parser.add_argument("--kd_mode", type=str, default="dmd2",
                            choices=["dmd2", "ctcd", "ctcd_ladd"],
                            help="Distillation mode: dmd2 (DMD2), ctcd (CTCD), ctcd_ladd (CTCD+LADD)")

        # DMD2 specific
        parser.add_argument("--dm_loss_weight", type=float, default=1.0,
                            help="Weight for distribution matching loss (DMD2)")
        parser.add_argument("--gan_loss_weight", type=float, default=1.0,
                            help="Weight for GAN loss (DMD2)")
        parser.add_argument("--real_guidance_scale", type=float, default=6.0,
                            help="CFG scale for real score estimation (DMD2)")
        parser.add_argument("--dfake_gen_update_ratio", type=int, default=5,
                            help="Ratio of D/G updates: D updated every step, G every N steps (DMD2)")

        # CTCD specific
        parser.add_argument("--sigma_data", type=float, default=1.0,
                            help="Data standard deviation for trigonometric flow (CTCD)")
        parser.add_argument("--tangent_warmup_steps", type=int, default=1000,
                            help="Warmup steps for tangent in CTCD")
        parser.add_argument("--ctcd_logit_mean", type=float, default=0.0,
                            help="Mean for logit-normal timestep sampling")
        parser.add_argument("--ctcd_logit_std", type=float, default=1.0,
                            help="Std for logit-normal timestep sampling")

        # LADD specific
        parser.add_argument("--adv_lambda", type=float, default=0.1,
                            help="Weight for adversarial loss (LADD)")
        parser.add_argument("--scm_lambda", type=float, default=1.0,
                            help="Weight for consistency loss when combined with LADD")
        parser.add_argument("--ladd_loss_type", type=str, default="hinge",
                            choices=["hinge", "cross_entropy"],
                            help="Discriminator loss type for LADD")

        # General training
        parser.add_argument("--learning_rate", type=float, default=1e-5,
                            help="Learning rate for generator/student")
        parser.add_argument("--guidance_lr", type=float, default=None,
                            help="Learning rate for guidance/discriminator (defaults to --learning_rate)")
        parser.add_argument("--num_epochs", type=int, default=5,
                            help="Number of training epochs")
        parser.add_argument("--batch_size", type=int, default=1,
                            help="Batch size for training")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                            help="Gradient accumulation steps")
        parser.add_argument("--use_ema", action="store_true", default=True,
                            help="Use Exponential Moving Average for student")

        # Dataset configuration
        parser.add_argument("--dataset_name", type=str, default="MSCOCO2017",
                            help="Dataset name: MSCOCO2017 or Flickr8k")
        parser.add_argument("--num_images", type=int, default=10000,
                            help="Number of captions to use for training")
        parser.add_argument("--coco_splits", type=str, default="auto",
                            choices=["val", "train", "both", "auto"])

        # Generation / evaluation
        parser.add_argument("--steps", type=int, default=4,
                            help="Number of inference steps for evaluation (1-4 for distilled models)")
        parser.add_argument("--guidance_scale", type=float, default=0.0,
                            help="Guidance scale (0 for distilled models, as CFG is baked in)")
        parser.add_argument("--skip_metrics", action="store_true",
                            help="Skip quality metrics calculation")
        parser.add_argument("--metrics_subset", type=int, default=100,
                            help="Number of images for metrics calculation")
        parser.add_argument("--skip_training", action="store_true",
                            help="Skip training, only run evaluation with existing checkpoint")
        parser.add_argument("--checkpoint_path", type=str, default=None,
                            help="Path to existing student checkpoint to load")
        parser.add_argument("--memory_efficient", action="store_true",
                            help="Enable memory-efficient mode: offload teacher/EMA/fakescore to CPU between steps. "
                                 "Slower but fits in ~15GB VRAM for SDXL DMD2.")

        args = parser.parse_args()

    # ---- Resolve parameters ----
    teacher_model = getattr(args, "teacher_model", None) or getattr(args, "model_name", "SDXL")
    student_model = getattr(args, "student_model", None)
    student_num_blocks = getattr(args, "student_num_blocks", None)
    kd_mode = getattr(args, "kd_mode", "dmd2")
    dm_loss_weight = getattr(args, "dm_loss_weight", 1.0)
    gan_loss_weight = getattr(args, "gan_loss_weight", 1.0)
    real_guidance_scale = getattr(args, "real_guidance_scale", 6.0)
    dfake_gen_update_ratio = getattr(args, "dfake_gen_update_ratio", 5)
    sigma_data = getattr(args, "sigma_data", 1.0)
    tangent_warmup_steps = getattr(args, "tangent_warmup_steps", 1000)
    ctcd_logit_mean = getattr(args, "ctcd_logit_mean", 0.0)
    ctcd_logit_std = getattr(args, "ctcd_logit_std", 1.0)
    adv_lambda = getattr(args, "adv_lambda", 0.1)
    scm_lambda = getattr(args, "scm_lambda", 1.0)
    ladd_loss_type = getattr(args, "ladd_loss_type", "hinge")
    learning_rate = getattr(args, "learning_rate", 1e-5)
    guidance_lr = getattr(args, "guidance_lr", None)
    num_epochs = getattr(args, "num_epochs", 5)
    batch_size = getattr(args, "batch_size", 1)
    gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 4)
    use_ema = getattr(args, "use_ema", True)
    dataset_name = getattr(args, "dataset_name", "MSCOCO2017")
    num_images = getattr(args, "num_images", 10000)
    steps = getattr(args, "steps", getattr(args, "inference_steps", 4))
    guidance_scale = getattr(args, "guidance_scale", 0.0)
    skip_metrics = getattr(args, "skip_metrics", False)
    metrics_subset = getattr(args, "metrics_subset", 100)
    skip_training = getattr(args, "skip_training", False)
    checkpoint_path = getattr(args, "checkpoint_path", None)
    memory_efficient = getattr(args, "memory_efficient", False)

    # ---- Output directory ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        "distillation", "outputs",
        f"{teacher_model}_{kd_mode}_kd_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  KNOWLEDGE DISTILLATION FOR DIFFUSION MODELS")
    print("  Methods: DMD2 (NeurIPS 2024) / SANA-Sprint CTCD+LADD (2025)")
    print("=" * 70)
    print(f"  Teacher model   : {teacher_model}")
    print(f"  Student model   : {student_model or teacher_model}")
    print(f"  KD mode         : {kd_mode}")
    print(f"  Dataset         : {dataset_name}")
    print(f"  Num images      : {num_images}")
    print(f"  Memory efficient: {memory_efficient}")
    print(f"  Output dir      : {output_dir}")
    print()

    # ---- Initialize pipeline ----
    kd_pipeline = KnowledgeDistillationPipeline(
        teacher_model_name=teacher_model,
        student_model_name=student_model,
        student_num_blocks=student_num_blocks,
        kd_mode=kd_mode,
        dm_loss_weight=dm_loss_weight,
        gan_loss_weight=gan_loss_weight,
        real_guidance_scale=real_guidance_scale,
        dfake_gen_update_ratio=dfake_gen_update_ratio,
        sigma_data=sigma_data,
        tangent_warmup_steps=tangent_warmup_steps,
        ctcd_logit_mean=ctcd_logit_mean,
        ctcd_logit_std=ctcd_logit_std,
        adv_lambda=adv_lambda,
        scm_lambda=scm_lambda,
        ladd_loss_type=ladd_loss_type,
        learning_rate=learning_rate,
        guidance_lr=guidance_lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_ema=use_ema,
        memory_efficient=memory_efficient,
    )

    # ---- Load teacher ----
    kd_pipeline.load_teacher()

    # ---- Create student ----
    kd_pipeline.create_student()

    # ---- Print compression stats ----
    stats = kd_pipeline.get_compression_stats()
    print("\n  Compression Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v:,}")

    # ---- Load dataset ----
    print("\n=== Loading Dataset ===")
    _shared = _import_shared()
    captions_dict = {}
    image_dimensions = {}
    original_dir = ""

    if dataset_name == "Flickr8k":
        flickr_dir = "flickr8k"
        caption_path = os.path.join(flickr_dir, "captions.txt")
        original_dir = os.path.join(flickr_dir, "Images")
        captions_dict, image_dimensions = _shared["process_flickr8k"](original_dir, caption_path, limit=num_images)
    else:
        coco_dir = "coco"
        original_dir = os.path.join(coco_dir, "val2017")
        coco_splits = getattr(args, "coco_splits", "auto")
        captions_dict, image_dimensions = _shared["process_coco_extended"](
            coco_dir, limit=num_images, coco_splits=coco_splits
        )

    captions_list = list(captions_dict.values())
    print(f"  Loaded {len(captions_list)} captions")

    # ---- Training ----
    if not skip_training:
        if checkpoint_path:
            kd_pipeline.load_student_checkpoint(checkpoint_path)
        training_log = kd_pipeline.train(captions_list, output_dir)
    else:
        if checkpoint_path:
            kd_pipeline.load_student_checkpoint(checkpoint_path)
        else:
            print("  Skipping training (no checkpoint provided, using initialized student)")

    # ---- Image Generation ----
    gen_dir = os.path.join(output_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    generation_times = kd_pipeline.generate_images(
        captions_dict, gen_dir,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    )

    # ---- Evaluation ----
    if not skip_metrics:
        metrics = kd_pipeline.evaluate(
            gen_dir, original_dir, captions_dict,
            image_dimensions=image_dimensions,
            metrics_subset=metrics_subset,
        )

    # ---- Save summary ----
    summary = {
        "teacher_model": teacher_model,
        "student_model": student_model or teacher_model,
        "kd_mode": kd_mode,
        "num_epochs": num_epochs,
        "dataset": dataset_name,
        "num_images": num_images,
        "compression_stats": stats,
        "output_dir": output_dir,
    }

    if kd_mode == "dmd2":
        summary.update({
            "dm_loss_weight": dm_loss_weight,
            "gan_loss_weight": gan_loss_weight,
            "dfake_gen_update_ratio": dfake_gen_update_ratio,
        })
    elif kd_mode in ("ctcd", "ctcd_ladd"):
        summary.update({
            "sigma_data": sigma_data,
            "tangent_warmup_steps": tangent_warmup_steps,
        })
        if kd_mode == "ctcd_ladd":
            summary.update({
                "adv_lambda": adv_lambda,
                "scm_lambda": scm_lambda,
            })

    summary_path = os.path.join(output_dir, "distillation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to {summary_path}")
    print("  Knowledge Distillation complete!")

    # Cleanup
    free_memory()


if __name__ == "__main__":
    main()
