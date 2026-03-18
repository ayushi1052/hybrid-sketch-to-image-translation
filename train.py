"""
train.py
=========
SGLDv2 Training Script

What gets trained : SketchAdapter only (~22M params)
What stays frozen : SD VAE + UNet + CLIP text encoder (100% frozen)

Loss = Diffusion MSE + λ * CLIP Perceptual Loss

Usage
-----
  python train.py --data_dir ./dataset --output_dir ./checkpoints --epochs 20

  # Quick test (no structure extraction)
  python train.py --data_dir ./dataset --no_structure --epochs 3

  # Specific categories only
  python train.py --data_dir ./dataset --categories aeroplane apple ball
"""

import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup

from models          import SGLDv2Model, StructureExtractor
from dataset_loader  import build_dataloader
from utils           import (
    set_seed, AverageMeter, log_config,
    save_sample_grid, count_parameters,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP Perceptual Loss
# ─────────────────────────────────────────────────────────────────────────────

class CLIPPerceptualLoss(torch.nn.Module):
    """
    Semantic consistency loss in CLIP's embedding space.
    Forces generated images to be semantically similar to ground truth
    beyond pixel-level MSE — this is what D-Sketch lacks.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        try:
            import clip
            self.model, _ = clip.load("ViT-B/32", device=device)
            self.model.requires_grad_(False)
            self.enabled = True
            print("  [CLIPLoss] CLIP perceptual loss: ENABLED")
        except ImportError:
            self.enabled = False
            print("  [CLIPLoss] clip not installed — using MSE only. "
                  "pip install clip-by-openai to enable.")

    def forward(
        self,
        generated: torch.Tensor,
        target:    torch.Tensor,
    ) -> torch.Tensor:
        """
        generated, target : (B, 3, H, W) float [-1, 1]
        returns: scalar CLIP cosine distance
        """
        if not self.enabled:
            return torch.tensor(0.0, device=generated.device)

        # CLIP expects [0,1] float, resized to 224x224
        g = F.interpolate((generated * 0.5 + 0.5).clamp(0,1), (224,224))
        t = F.interpolate((target    * 0.5 + 0.5).clamp(0,1), (224,224))

        g_feat = self.model.encode_image(g)
        t_feat = self.model.encode_image(t)

        return 1 - F.cosine_similarity(g_feat, t_feat).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SGLDv2 SketchAdapter")

    # Paths
    p.add_argument("--data_dir",    type=str, required=True,
                   help="Dataset root (with sketch/ and photo/ subfolders)")
    p.add_argument("--output_dir",  type=str, default="./checkpoints")
    p.add_argument("--resume",      type=str, default=None,
                   help="Path to SketchAdapter checkpoint to resume from")

    # Model
    p.add_argument("--base_model",  type=str,
                   default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--num_tokens",  type=int, default=16,
                   help="Number of sketch tokens (more = richer conditioning)")
    p.add_argument("--attn_scale",  type=float, default=1.0,
                   help="Scale of sketch attention output (1.0 recommended)")

    # Dataset
    p.add_argument("--img_size",    type=int, default=256,
                   help="Image size (256 for your dataset)")
    p.add_argument("--categories",  type=str, nargs="+", default=None,
                   help="Specific categories to train on (default: all)")
    p.add_argument("--no_augment",  action="store_true")
    p.add_argument("--no_structure",action="store_true",
                   help="Skip structure extraction (faster, for testing)")

    # Training
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Higher LR than ControlNet (fewer params to train)")
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int,   default=200)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--clip_weight",  type=float, default=0.1,
                   help="Weight for CLIP perceptual loss (0 = disable)")

    # Infrastructure
    p.add_argument("--num_workers",     type=int, default=2)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="fp16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--save_every",  type=int, default=2)
    p.add_argument("--sample_every",type=int, default=1)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive timestep sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_timesteps_adaptive(
    batch_size: int,
    scheduler,
    structure_maps: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Adaptive timestep sampling — assigns larger timesteps to more
    abstract (sparse) sketches, which have more ambiguity to resolve.

    D-Sketch uses uniform sampling; this gives us better training signal
    for the wide variety of sketch styles in your dataset.
    """
    T = scheduler.config.num_train_timesteps

    # Estimate sketch complexity from edge channel (ch 0)
    edge_density = structure_maps[:, 0].mean(dim=[1, 2])    # (B,) in [0,1]
    # More edges = more detail = lower timesteps preferred
    # Fewer edges = more abstract = higher timesteps preferred
    alpha = 1.0 + (1.0 - edge_density.cpu()) * 3.0         # [1, 4]

    timesteps = []
    for a in alpha:
        probs = torch.arange(1, T+1, dtype=torch.float) ** float(a)
        probs = probs / probs.sum()
        t = torch.multinomial(probs, 1).item()
        timesteps.append(t)

    return torch.tensor(timesteps, device=device).long()


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def training_step(
    batch:       dict,
    model:       SGLDv2Model,
    clip_loss_fn,
    args:        argparse.Namespace,
    device:      str,
) -> torch.Tensor:
    """Single forward + loss computation for one batch."""

    # 1. Text embeddings
    encoder_hidden = model.encode_text(list(batch["caption"]))

    # 2. Encode photo → latent
    latents = model.encode_image(batch["photo"].to(device))

    # 3. Adaptive noise + timestep sampling
    noise      = torch.randn_like(latents)
    timesteps  = sample_timesteps_adaptive(
        latents.shape[0], model.noise_scheduler,
        batch["structure"], device
    )
    noisy_lat  = model.add_noise(latents, noise, timesteps)

    # 4. Forward pass (SketchAdapter injects sketch tokens into frozen UNet)
    noise_pred = model(
        noisy_latents  = noisy_lat,
        timesteps      = timesteps,
        encoder_hidden = encoder_hidden,
        structure_maps = batch["structure"].to(device),
    )

    # 5. Diffusion MSE loss
    diff_loss = F.mse_loss(noise_pred.float(), noise.float())

    # 6. CLIP perceptual loss on partially denoised predictions
    total_loss = diff_loss
    if args.clip_weight > 0:
        # Decode noisy prediction for CLIP comparison (approximate)
        with torch.no_grad():
            pred_latents = model.noise_scheduler.step(
                noise_pred.float(), timesteps[0], noisy_lat.float()
            ).pred_original_sample
        decoded = model.decode_latents(pred_latents.to(model.dtype))
        clip_l  = clip_loss_fn(decoded.float(), batch["photo"].float().to(device))
        total_loss = diff_loss + args.clip_weight * clip_l

    return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision = args.mixed_precision,
        log_with        = "tensorboard",
        project_dir     = args.output_dir,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        log_config(vars(args), args.output_dir)
        print(f"\n{'='*60}")
        print("  SGLDv2 Training — SketchAdapter (SD fully frozen)")
        print(f"{'='*60}")
        print(f"  Device        : {device}")
        print(f"  Mixed prec.   : {args.mixed_precision}")
        print(f"  Epochs        : {args.epochs}")
        print(f"  Batch size    : {args.batch_size}")
        print(f"  LR            : {args.lr}")
        print(f"  Num tokens    : {args.num_tokens}")
        print(f"  CLIP weight   : {args.clip_weight}")
        print(f"{'='*60}\n")

    # ── Structure extractor ────────────────────────────────────────────────
    extractor = None if args.no_structure else StructureExtractor(device=device)

    # ── Dataset ───────────────────────────────────────────────────────────
    print("[1/4] Building dataset …")
    dataloader = build_dataloader(
        data_dir    = args.data_dir,
        img_size    = args.img_size,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        extractor   = extractor,
        categories  = args.categories,
        augment     = not args.no_augment,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("[2/4] Loading SGLDv2 model …")
    model = SGLDv2Model(
        base_model  = args.base_model,
        img_size    = args.img_size,
        num_tokens  = args.num_tokens,
        attn_scale  = args.attn_scale,
        device      = device,
    )
    if args.resume:
        model.load_adapter(args.resume)
        print(f"  Resumed from: {args.resume}")

    if accelerator.is_main_process:
        params = sum(p.numel() for p in model.trainable_parameters())
        print(f"  Trainable params: {params:,}  (~{params/1e6:.1f}M)")

    # ── CLIP loss ──────────────────────────────────────────────────────────
    clip_loss_fn = CLIPPerceptualLoss(device=device)

    # ── Optimiser + LR scheduler ───────────────────────────────────────────
    print("[3/4] Setting up optimiser …")
    optimiser   = torch.optim.AdamW(
        model.trainable_parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.999),
    )
    total_steps  = len(dataloader) * args.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimiser,
        num_warmup_steps   = args.warmup_steps,
        num_training_steps = total_steps,
    )

    # ── Accelerate ─────────────────────────────────────────────────────────
    (
        model.sketch_adapter,
        optimiser,
        dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model.sketch_adapter,
        optimiser,
        dataloader,
        lr_scheduler,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    print("[4/4] Starting training …\n")
    global_step = 0
    best_loss   = float("inf")

    for epoch in range(1, args.epochs + 1):
        meter = AverageMeter()
        bar   = tqdm(
            dataloader,
            desc    = f"Epoch {epoch:03d}/{args.epochs:03d}",
            disable = not accelerator.is_local_main_process,
        )

        for batch in bar:
            with accelerator.accumulate(model.sketch_adapter):
                loss = training_step(batch, model, clip_loss_fn, args, device)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.trainable_parameters(), args.grad_clip
                    )

                optimiser.step()
                lr_scheduler.step()
                optimiser.zero_grad()

            meter.update(loss.item())
            global_step += 1
            bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg":  f"{meter.avg:.4f}",
                "lr":   f"{lr_scheduler.get_last_lr()[0]:.2e}",
            })

        # ── End of epoch ───────────────────────────────────────────────────
        if accelerator.is_main_process:
            avg = meter.avg
            print(f"  Epoch {epoch:03d} | loss={avg:.4f}")
            accelerator.log({"loss": avg, "lr": lr_scheduler.get_last_lr()[0]},
                            step=global_step)

            if epoch % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"epoch_{epoch:03d}")
                model.save(ckpt)
                print(f"  ✓ Checkpoint → {ckpt}")

            if avg < best_loss:
                best_loss = avg
                model.save(os.path.join(args.output_dir, "best"))
                print(f"  ★ Best (loss={best_loss:.4f}) → checkpoints/best")

            if epoch % args.sample_every == 0:
                sp = os.path.join(args.output_dir, "samples", f"ep{epoch:03d}.png")
                os.makedirs(os.path.dirname(sp), exist_ok=True)
                save_sample_grid(batch, sp)

    # ── Final save ─────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        model.save(os.path.join(args.output_dir, "final"))
        print(f"\n Training complete.")
        print(f"  Best loss: {best_loss:.4f}")
        print(f"  Checkpoints: {args.output_dir}")


if __name__ == "__main__":
    train()
