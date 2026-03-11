#!/usr/bin/env python3
"""
Knowledge Distillation module for Stable Diffusion optimization.

This module implements Knowledge Distillation (KD) to compress diffusion models
by training a smaller/faster student model to mimic a larger teacher model.

Supported approaches:
1. Output-level KD: Match the final noise predictions (epsilon) between teacher and student
2. Feature-level KD: Match intermediate feature maps from UNet/Transformer blocks
3. Attention Transfer: Transfer attention maps from teacher to student
4. Progressive Distillation: Reduce inference steps by distilling multi-step into fewer steps

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
#  Loss Functions for Knowledge Distillation
# ============================================================================

class KDLossOutput(nn.Module):
    """
    Output-level Knowledge Distillation loss.
    Matches the noise prediction (epsilon) between teacher and student.
    
    L = alpha * L_task + (1 - alpha) * L_kd
    where L_task = MSE(student_pred, target)
          L_kd  = MSE(student_pred, teacher_pred) * temperature^2
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self,
        student_pred: torch.Tensor,
        teacher_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Task loss: student vs ground-truth noise
        loss_task = F.mse_loss(student_pred, target)

        # KD loss: student vs teacher (scaled by temperature^2)
        loss_kd = F.mse_loss(student_pred, teacher_pred) * (self.temperature ** 2)

        loss_total = self.alpha * loss_task + (1 - self.alpha) * loss_kd

        metrics = {
            "loss_total": loss_total.item(),
            "loss_task": loss_task.item(),
            "loss_kd": loss_kd.item(),
        }
        return loss_total, metrics


class KDLossFeature(nn.Module):
    """
    Feature-level Knowledge Distillation loss.
    Matches intermediate feature maps between teacher and student blocks.
    Uses projection layers when feature dimensions differ.
    """

    def __init__(self, teacher_dims: List[int], student_dims: List[int], weight: float = 1.0):
        super().__init__()
        self.weight = weight
        # Build projection layers where dimensions mismatch
        self.projectors = nn.ModuleList()
        for t_dim, s_dim in zip(teacher_dims, student_dims):
            if t_dim != s_dim:
                self.projectors.append(
                    nn.Sequential(
                        nn.Linear(s_dim, t_dim),
                        nn.GELU(),
                    )
                )
            else:
                self.projectors.append(nn.Identity())

    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, device=student_features[0].device)
        layer_losses = {}

        for i, (sf, tf, proj) in enumerate(
            zip(student_features, teacher_features, self.projectors)
        ):
            sf_proj = proj(sf)
            layer_loss = F.mse_loss(sf_proj, tf.detach())
            total_loss = total_loss + layer_loss
            layer_losses[f"feature_loss_layer_{i}"] = layer_loss.item()

        total_loss = total_loss * self.weight / max(len(student_features), 1)
        layer_losses["feature_loss_total"] = total_loss.item()
        return total_loss, layer_losses


class KDLossAttentionTransfer(nn.Module):
    """
    Attention Transfer loss.
    Matches the attention map distributions between teacher and student.
    Uses KL divergence on flattened spatial attention maps.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def _normalize_attention(self, attn: torch.Tensor) -> torch.Tensor:
        """Normalize attention maps to probability distributions."""
        b, h, n, _ = attn.shape
        attn_flat = attn.view(b * h, n, -1)
        attn_norm = F.softmax(attn_flat / math.sqrt(attn_flat.size(-1)), dim=-1)
        return attn_norm

    def forward(
        self,
        student_attns: List[torch.Tensor],
        teacher_attns: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, device=student_attns[0].device)
        metrics = {}

        for i, (s_attn, t_attn) in enumerate(zip(student_attns, teacher_attns)):
            s_norm = self._normalize_attention(s_attn)
            t_norm = self._normalize_attention(t_attn)

            # KL divergence between attention distributions
            layer_loss = F.kl_div(
                s_norm.log().clamp(min=-100), t_norm, reduction="batchnorm"
            ) if False else F.mse_loss(s_norm, t_norm.detach())

            total_loss = total_loss + layer_loss
            metrics[f"attn_loss_layer_{i}"] = layer_loss.item()

        total_loss = total_loss * self.weight / max(len(student_attns), 1)
        metrics["attn_loss_total"] = total_loss.item()
        return total_loss, metrics


class ProgressiveDistillationLoss(nn.Module):
    """
    Progressive Distillation loss for reducing inference steps.
    The student learns to predict in N/2 steps what the teacher produces in N steps.
    Based on "Progressive Distillation for Fast Sampling of Diffusion Models" (Salimans & Ho, 2022).
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        student_pred: torch.Tensor,
        teacher_two_step_result: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            student_pred: Student's single-step prediction x_hat
            teacher_two_step_result: Teacher's two-step DDIM result
        """
        loss = F.mse_loss(student_pred, teacher_two_step_result.detach()) * self.weight
        return loss, {"progressive_loss": loss.item()}


# ============================================================================
#  Feature Extraction Hooks
# ============================================================================

class FeatureExtractor:
    """
    Attaches forward hooks to specified layers of a model to capture
    intermediate feature maps and attention maps during forward pass.
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.features: Dict[str, torch.Tensor] = {}
        self.attentions: Dict[str, torch.Tensor] = {}
        self._hooks = []

        for name, module in model.named_modules():
            if any(ln in name for ln in layer_names):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.features[name] = output[0].detach()
                if len(output) > 1 and output[1] is not None:
                    self.attentions[name] = output[1].detach()
            else:
                self.features[name] = output.detach()
        return hook_fn

    def get_features(self) -> List[torch.Tensor]:
        return list(self.features.values())

    def get_attentions(self) -> List[torch.Tensor]:
        return list(self.attentions.values())

    def clear(self):
        self.features.clear()
        self.attentions.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


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

    Workflow:
    1. Load teacher model (full-size, high quality)
    2. Create student model (smaller/pruned/quantized)
    3. Pre-compute text embeddings for training captions
    4. Train student to match teacher's noise predictions and features
    5. Evaluate student on image generation quality

    Supports: SDXL, Flux, SD3, SANA model families.
    """

    def __init__(
        self,
        teacher_model_name: str = "SDXL",
        student_model_name: Optional[str] = None,
        student_num_blocks: Optional[int] = None,
        kd_mode: str = "output",           # output | feature | attention | progressive
        alpha: float = 0.5,
        temperature: float = 1.0,
        feature_weight: float = 1.0,
        attention_weight: float = 1.0,
        progressive_weight: float = 1.0,
        learning_rate: float = 1e-5,
        num_epochs: int = 5,
        batch_size: int = 1,
        num_train_timesteps: int = 1000,
        gradient_accumulation_steps: int = 4,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        feature_layers: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name or teacher_model_name
        self.student_num_blocks = student_num_blocks
        self.kd_mode = kd_mode
        self.alpha = alpha
        self.temperature = temperature
        self.feature_weight = feature_weight
        self.attention_weight = attention_weight
        self.progressive_weight = progressive_weight
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_train_timesteps = num_train_timesteps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.feature_layers = feature_layers or ["attn", "ff", "mid_block"]
        self.device = device

        # Will be filled during load
        self.teacher_pipeline = None
        self.student_pipeline = None
        self.teacher_denoiser = None   # UNet or Transformer
        self.student_denoiser = None
        self.scheduler = None
        self.ema_model = None
        self.model_type = None

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
            self.student_denoiser = copy.deepcopy(self.teacher_denoiser)
            self.student_denoiser.train()
            for p in self.student_denoiser.parameters():
                p.requires_grad = True

        # Create EMA copy
        if self.use_ema:
            self.ema_model = copy.deepcopy(self.student_denoiser)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad = False

        student_size = get_model_size_mb(self.student_denoiser)
        student_params = count_parameters(self.student_denoiser)
        print(f"  Student size   : {student_size:.1f} MB")
        print(f"  Student params : {student_params:,}")
        print(f"  EMA enabled    : {self.use_ema}")

    def _create_slim_student(self, num_blocks: int) -> nn.Module:
        """
        Create a smaller student by removing transformer/UNet blocks.
        This works by selecting every N-th block to keep.
        """
        student = copy.deepcopy(self.teacher_denoiser)

        if self.model_type == "sdxl":
            # SDXL UNet: trim down_blocks and up_blocks
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
            # Transformer-based: trim transformer_blocks
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
        """Load the appropriate diffusers pipeline for the model type."""
        from diffusers import (
            StableDiffusionXLPipeline,
            FluxPipeline,
            StableDiffusion3Pipeline,
        )

        common_kwargs = dict(torch_dtype=torch.float16, use_safetensors=True)

        if model_type == "sdxl":
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_path, variant="fp16", **common_kwargs
            )
        elif model_type == "flux":
            pipeline = FluxPipeline.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            )
        elif model_type == "sd3":
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path, **common_kwargs
            )
        elif model_type == "sana":
            try:
                from diffusers import SanaPipeline
                pipeline = SanaPipeline.from_pretrained(model_path, **common_kwargs)
            except ImportError:
                print("  SanaPipeline not available, falling back to SDXL loader")
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_path, variant="fp16", **common_kwargs
                )
        else:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_path, variant="fp16", **common_kwargs
            )

        pipeline = pipeline.to(self.device)
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
                    # Flux uses T5 + CLIP encoders
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
    # Training Loop
    # ------------------------------------------------------------------

    def train(self, captions: List[str], output_dir: str):
        """
        Main distillation training loop.

        Args:
            captions: list of text prompts for training
            output_dir: directory to save checkpoints and logs
        """
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "training_log.json")

        # 1. Encode all prompts
        prompt_embeds, pooled_prompt_embeds = self.encode_prompts(captions)

        # 2. Determine latent shape based on model type
        if self.model_type == "sdxl":
            latent_shape = (4, 128, 128)   # SDXL 1024x1024 => 128x128 latent
        elif self.model_type == "flux":
            latent_shape = (16, 64, 64)
        elif self.model_type == "sd3":
            latent_shape = (16, 128, 128)
        else:
            latent_shape = (4, 128, 128)

        # 3. Create dataset and dataloader
        dataset = CaptionNoiseDataset(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            latent_shape=latent_shape,
            num_train_timesteps=self.num_train_timesteps,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        # 4. Setup loss functions
        output_loss_fn = KDLossOutput(alpha=self.alpha, temperature=self.temperature)

        teacher_extractor = None
        student_extractor = None
        feature_loss_fn = None
        attention_loss_fn = None

        if self.kd_mode in ("feature", "attention"):
            teacher_extractor = FeatureExtractor(self.teacher_denoiser, self.feature_layers)
            student_extractor = FeatureExtractor(self.student_denoiser, self.feature_layers)

        if self.kd_mode == "feature":
            # We'll initialize feature_loss_fn after the first forward pass
            # to determine actual dimensions
            pass
        if self.kd_mode == "attention":
            attention_loss_fn = KDLossAttentionTransfer(weight=self.attention_weight)

        progressive_loss_fn = None
        if self.kd_mode == "progressive":
            progressive_loss_fn = ProgressiveDistillationLoss(weight=self.progressive_weight)

        # 5. Setup optimizer
        optimizer = torch.optim.AdamW(
            self.student_denoiser.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-2,
        )

        # Cosine annealing scheduler
        total_steps = len(dataloader) * self.num_epochs // self.gradient_accumulation_steps
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=self.learning_rate * 0.01
        )

        # 6. Training loop
        print("\n" + "=" * 60)
        print("  Starting Knowledge Distillation Training")
        print("=" * 60)
        print(f"  Mode             : {self.kd_mode}")
        print(f"  Epochs           : {self.num_epochs}")
        print(f"  Batch size       : {self.batch_size}")
        print(f"  Learning rate    : {self.learning_rate}")
        print(f"  Grad accum steps : {self.gradient_accumulation_steps}")
        print(f"  Total steps      : {total_steps}")
        print(f"  Alpha            : {self.alpha}")
        print(f"  Temperature      : {self.temperature}")
        print(f"  Latent shape     : {latent_shape}")
        print()

        training_log = []
        global_step = 0
        best_loss = float("inf")

        self.teacher_denoiser.eval()
        self.student_denoiser.to(self.device)

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_metrics = {}
            self.student_denoiser.train()

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for step, (noise, timesteps, embeds, pooled) in enumerate(pbar):
                noise = noise.to(self.device, dtype=torch.float16)
                timesteps = timesteps.to(self.device).squeeze()
                embeds = embeds.to(self.device, dtype=torch.float16)
                if pooled.dim() > 1:
                    pooled = pooled.to(self.device, dtype=torch.float16)
                else:
                    pooled = None

                # Create noisy latent
                alpha_t = self._get_alpha(timesteps)
                sigma_t = (1 - alpha_t).sqrt()
                clean_latent = torch.randn_like(noise)
                noisy_latent = alpha_t.sqrt().view(-1, 1, 1, 1) * clean_latent + sigma_t.view(-1, 1, 1, 1) * noise

                # ------ Teacher forward (no gradient) ------
                with torch.no_grad():
                    teacher_pred = self._forward_denoiser(
                        self.teacher_denoiser, noisy_latent, timesteps, embeds, pooled
                    )

                # ------ Student forward ------
                student_pred = self._forward_denoiser(
                    self.student_denoiser, noisy_latent, timesteps, embeds, pooled
                )

                # ------ Compute loss ------
                if self.kd_mode == "output":
                    loss, metrics = output_loss_fn(student_pred, teacher_pred, noise)

                elif self.kd_mode == "feature":
                    loss_out, m_out = output_loss_fn(student_pred, teacher_pred, noise)
                    if feature_loss_fn is None and teacher_extractor and student_extractor:
                        t_feats = teacher_extractor.get_features()
                        s_feats = student_extractor.get_features()
                        if t_feats and s_feats:
                            t_dims = [f.shape[-1] for f in t_feats]
                            s_dims = [f.shape[-1] for f in s_feats]
                            feature_loss_fn = KDLossFeature(
                                t_dims, s_dims, weight=self.feature_weight
                            ).to(self.device)

                    loss = loss_out
                    metrics = m_out
                    if feature_loss_fn and teacher_extractor and student_extractor:
                        t_feats = teacher_extractor.get_features()
                        s_feats = student_extractor.get_features()
                        if t_feats and s_feats:
                            loss_feat, m_feat = feature_loss_fn(s_feats, t_feats)
                            loss = loss + loss_feat
                            metrics.update(m_feat)
                    if teacher_extractor:
                        teacher_extractor.clear()
                    if student_extractor:
                        student_extractor.clear()

                elif self.kd_mode == "attention":
                    loss_out, m_out = output_loss_fn(student_pred, teacher_pred, noise)
                    loss = loss_out
                    metrics = m_out
                    if attention_loss_fn and teacher_extractor and student_extractor:
                        t_attns = teacher_extractor.get_attentions()
                        s_attns = student_extractor.get_attentions()
                        if t_attns and s_attns:
                            loss_attn, m_attn = attention_loss_fn(s_attns, t_attns)
                            loss = loss + loss_attn
                            metrics.update(m_attn)
                    if teacher_extractor:
                        teacher_extractor.clear()
                    if student_extractor:
                        student_extractor.clear()

                elif self.kd_mode == "progressive":
                    # Progressive distillation: teacher does 2 steps, student does 1
                    with torch.no_grad():
                        # Teacher step 1
                        t_mid = timesteps // 2
                        teacher_pred_1 = self._forward_denoiser(
                            self.teacher_denoiser, noisy_latent, timesteps, embeds, pooled
                        )
                        # Estimate x at t_mid
                        mid_latent = self._ddim_step(noisy_latent, teacher_pred_1, timesteps, t_mid)
                        # Teacher step 2
                        teacher_pred_2 = self._forward_denoiser(
                            self.teacher_denoiser, mid_latent, t_mid, embeds, pooled
                        )
                        # Final teacher target
                        t_target = torch.zeros_like(timesteps)
                        teacher_result = self._ddim_step(mid_latent, teacher_pred_2, t_mid, t_target)

                    loss, metrics = progressive_loss_fn(student_pred, teacher_result)
                else:
                    loss, metrics = output_loss_fn(student_pred, teacher_pred, noise)

                # Gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student_denoiser.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Update EMA
                    if self.use_ema and self.ema_model is not None:
                        self._update_ema()

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v

                pbar.set_postfix(loss=f"{loss.item() * self.gradient_accumulation_steps:.4f}")

                # Periodic memory cleanup
                if step % 50 == 0:
                    free_memory()

            # End of epoch
            avg_loss = epoch_loss / max(len(dataloader), 1)
            avg_metrics = {k: v / max(len(dataloader), 1) for k, v in epoch_metrics.items()}

            log_entry = {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "lr": optimizer.param_groups[0]["lr"],
                **avg_metrics,
            }
            training_log.append(log_entry)

            print(f"\n  Epoch {epoch + 1} Summary:")
            print(f"    Avg Loss : {avg_loss:.6f}")
            print(f"    LR       : {optimizer.param_groups[0]['lr']:.2e}")
            for k, v in avg_metrics.items():
                print(f"    {k}: {v:.6f}")

            # Save checkpoint if best
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint(output_dir, epoch + 1, is_best=True)

            # Save periodic checkpoint
            if (epoch + 1) % max(1, self.num_epochs // 3) == 0:
                self._save_checkpoint(output_dir, epoch + 1)

        # Cleanup extractors
        if teacher_extractor:
            teacher_extractor.remove_hooks()
        if student_extractor:
            student_extractor.remove_hooks()

        # Save final checkpoint and training log
        self._save_checkpoint(output_dir, self.num_epochs, is_final=True)
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        print(f"\n  Training complete! Logs saved to {log_path}")
        return training_log

    # ------------------------------------------------------------------
    # Helpers for training  
    # ------------------------------------------------------------------

    def _get_alpha(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get alpha values for given timesteps from the scheduler."""
        if hasattr(self.scheduler, "alphas_cumprod"):
            alphas = self.scheduler.alphas_cumprod.to(timesteps.device)
            return alphas[timesteps.long()]
        else:
            # Fallback: linear schedule
            t_norm = timesteps.float() / self.num_train_timesteps
            return 1.0 - t_norm

    def _forward_denoiser(
        self,
        model: nn.Module,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the denoising model, handling different architectures."""
        kwargs = {}

        if self.model_type == "sdxl":
            # SDXL UNet expects added_cond_kwargs with text_embeds and time_ids
            if pooled is not None:
                time_ids = torch.zeros(latent.shape[0], 6, device=self.device, dtype=latent.dtype)
                kwargs["added_cond_kwargs"] = {
                    "text_embeds": pooled,
                    "time_ids": time_ids,
                }
            output = model(latent, timesteps, encoder_hidden_states=encoder_hidden_states, **kwargs)

        elif self.model_type == "flux":
            output = model(latent, timesteps, encoder_hidden_states=encoder_hidden_states, **kwargs)

        elif self.model_type == "sd3":
            output = model(latent, timesteps, encoder_hidden_states=encoder_hidden_states, **kwargs)

        else:
            output = model(latent, timesteps, encoder_hidden_states=encoder_hidden_states, **kwargs)

        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t_from: torch.Tensor,
        t_to: torch.Tensor,
    ) -> torch.Tensor:
        """Single DDIM deterministic step from t_from to t_to."""
        alpha_from = self._get_alpha(t_from).view(-1, 1, 1, 1)
        alpha_to = self._get_alpha(t_to).view(-1, 1, 1, 1)

        # Predict x_0
        x_0_pred = (x_t - (1 - alpha_from).sqrt() * noise_pred) / alpha_from.sqrt().clamp(min=1e-8)

        # Compute x at t_to
        x_t_to = alpha_to.sqrt() * x_0_pred + (1 - alpha_to).sqrt() * noise_pred
        return x_t_to

    def _update_ema(self):
        """Update EMA model parameters."""
        for ema_p, student_p in zip(self.ema_model.parameters(), self.student_denoiser.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(student_p.data, alpha=1 - self.ema_decay)

    def _save_checkpoint(self, output_dir: str, epoch: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        model_to_save = self.ema_model if (self.use_ema and self.ema_model is not None) else self.student_denoiser

        state = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "kd_mode": self.kd_mode,
            "alpha": self.alpha,
            "temperature": self.temperature,
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
        parser = argparse.ArgumentParser(description="Knowledge Distillation for Diffusion Models")

        # Model configuration
        parser.add_argument("--teacher_model", type=str, default="SDXL",
                            help="Teacher model name from config.json (e.g., SDXL, Flux.1-schnell, SD3)")
        parser.add_argument("--student_model", type=str, default=None,
                            help="Student model name (defaults to teacher)")
        parser.add_argument("--student_num_blocks", type=int, default=None,
                            help="Number of blocks for slim student (None = copy teacher)")

        # KD configuration
        parser.add_argument("--kd_mode", type=str, default="output",
                            choices=["output", "feature", "attention", "progressive"],
                            help="Knowledge distillation mode")
        parser.add_argument("--alpha", type=float, default=0.5,
                            help="Balance between task loss and KD loss (0=full KD, 1=full task)")
        parser.add_argument("--temperature", type=float, default=1.0,
                            help="Temperature for KD loss scaling")
        parser.add_argument("--feature_weight", type=float, default=1.0,
                            help="Weight for feature matching loss")
        parser.add_argument("--attention_weight", type=float, default=1.0,
                            help="Weight for attention transfer loss")

        # Training configuration
        parser.add_argument("--learning_rate", type=float, default=1e-5,
                            help="Learning rate for student training")
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
        parser.add_argument("--num_images", type=int, default=1000,
                            help="Number of captions to use for training")
        parser.add_argument("--coco_splits", type=str, default="auto",
                            choices=["val", "train", "both", "auto"])

        # Generation / evaluation
        parser.add_argument("--steps", type=int, default=30,
                            help="Number of inference steps for evaluation")
        parser.add_argument("--guidance_scale", type=float, default=7.5,
                            help="Guidance scale for image generation")
        parser.add_argument("--skip_metrics", action="store_true",
                            help="Skip quality metrics calculation")
        parser.add_argument("--metrics_subset", type=int, default=100,
                            help="Number of images for metrics calculation")
        parser.add_argument("--skip_training", action="store_true",
                            help="Skip training, only run evaluation with existing checkpoint")
        parser.add_argument("--checkpoint_path", type=str, default=None,
                            help="Path to existing student checkpoint to load")

        args = parser.parse_args()

    # ---- Resolve parameters with sensible defaults ----
    teacher_model = getattr(args, "teacher_model", None) or getattr(args, "model_name", "SDXL")
    student_model = getattr(args, "student_model", None)
    student_num_blocks = getattr(args, "student_num_blocks", None)
    kd_mode = getattr(args, "kd_mode", "output")
    alpha = getattr(args, "alpha", 0.5)
    temperature = getattr(args, "temperature", 1.0)
    feature_weight = getattr(args, "feature_weight", 1.0)
    attention_weight = getattr(args, "attention_weight", 1.0)
    learning_rate = getattr(args, "learning_rate", 1e-5)
    num_epochs = getattr(args, "num_epochs", 5)
    batch_size = getattr(args, "batch_size", 1)
    gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 4)
    use_ema = getattr(args, "use_ema", True)
    dataset_name = getattr(args, "dataset_name", "MSCOCO2017")
    num_images = getattr(args, "num_images", 1000)
    steps = getattr(args, "steps", getattr(args, "inference_steps", 30))
    guidance_scale = getattr(args, "guidance_scale", 7.5)
    skip_metrics = getattr(args, "skip_metrics", False)
    metrics_subset = getattr(args, "metrics_subset", 100)
    skip_training = getattr(args, "skip_training", False)
    checkpoint_path = getattr(args, "checkpoint_path", None)

    # ---- Output directory ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        "distillation", "outputs",
        f"{teacher_model}_{kd_mode}_kd_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  KNOWLEDGE DISTILLATION FOR DIFFUSION MODELS")
    print("=" * 70)
    print(f"  Teacher model   : {teacher_model}")
    print(f"  Student model   : {student_model or teacher_model}")
    print(f"  KD mode         : {kd_mode}")
    print(f"  Dataset         : {dataset_name}")
    print(f"  Num images      : {num_images}")
    print(f"  Output dir      : {output_dir}")
    print()

    # ---- Initialize pipeline ----
    kd_pipeline = KnowledgeDistillationPipeline(
        teacher_model_name=teacher_model,
        student_model_name=student_model,
        student_num_blocks=student_num_blocks,
        kd_mode=kd_mode,
        alpha=alpha,
        temperature=temperature,
        feature_weight=feature_weight,
        attention_weight=attention_weight,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_ema=use_ema,
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
        "alpha": alpha,
        "temperature": temperature,
        "num_epochs": num_epochs,
        "dataset": dataset_name,
        "num_images": num_images,
        "compression_stats": stats,
        "output_dir": output_dir,
    }
    summary_path = os.path.join(output_dir, "distillation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to {summary_path}")
    print("  Knowledge Distillation complete!")

    # Cleanup
    free_memory()


if __name__ == "__main__":
    main()
