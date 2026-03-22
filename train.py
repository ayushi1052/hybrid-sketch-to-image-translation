"""
train.py
=========
SGLDv2 Training — StructureControlNet on frozen Stable Diffusion

What trains
───────────
  StructureInputProjection + ControlNet  ← only these update
  UNet / VAE / CLIP                      ← fully frozen

Optimisations in this script
─────────────────────────────
• Min-SNR-γ (gamma=5) loss weighting   → faster convergence, better quality
• EMA (decay 0.9999)                   → stable checkpoint, smoother outputs
• Gradient accumulation                → effective larger batches on small GPUs
• structure maps disk-cached after epoch 1 → fast subsequent epochs
• TF32 enabled on Ampere+ GPUs         → free ~10 % speedup

Usage
─────
  python train.py --data_dir ./dataset --epochs 20
  python train.py --data_dir ./dataset --epochs 3 --no_structure
  python train.py --data_dir ./dataset --resume ./checkpoints/latest --epochs 10
"""

from __future__ import annotations

import os
import sys
import argparse
import traceback
import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel

from models         import SGLDv2Model, StructureExtractor
from dataset_loader import build_dataloader, FIXED_CAPTION
from utils          import (
    set_seed, AverageMeter, log_config,
    save_sample_grid, count_parameters,
)


# ─────────────────────────────────────────────────────────────────────────────
# Optional CLIP perceptual loss
# ─────────────────────────────────────────────────────────────────────────────

class CLIPPerceptualLoss(torch.nn.Module):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        try:
            import clip
            self.model, _ = clip.load("ViT-B/32", device=device)
            self.model.requires_grad_(False)
            self.enabled = True
            print("  [CLIPLoss] CLIP perceptual loss: ENABLED")
        except Exception as e:
            self.enabled = False
            print(f"  [CLIPLoss] Not available ({e}) — MSE only.")

    def forward(self, gen: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.tensor(0.0, device=gen.device)
        try:
            g = F.interpolate((gen * 0.5 + 0.5).clamp(0, 1), (224, 224))
            t = F.interpolate((tgt * 0.5 + 0.5).clamp(0, 1), (224, 224))
            return 1.0 - F.cosine_similarity(
                self.model.encode_image(g),
                self.model.encode_image(t),
            ).mean()
        except Exception as e:
            print(f"  [CLIPLoss] Skipped: {e}")
            return torch.tensor(0.0, device=gen.device)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive timestep sampling (sketch-density aware)
# ─────────────────────────────────────────────────────────────────────────────

def sample_timesteps(
    batch_size:     int,
    scheduler,
    structure_maps: torch.Tensor,
    device:         str,
) -> torch.Tensor:
    """
    Bias timestep distribution toward noisier steps for sparse sketches
    (low edge density → model needs to learn stronger structure priors).
    Falls back to uniform sampling on any error.
    """
    try:
        T    = scheduler.config.num_train_timesteps
        edge = structure_maps[:, 0].mean(dim=[1, 2]).cpu()   # ch0 = edge
        alpha = 1.0 + (1.0 - edge) * 3.0
        ts    = []
        for a in alpha:
            p = torch.arange(1, T + 1, dtype=torch.float) ** float(a)
            p = p / p.sum()
            ts.append(torch.multinomial(p, 1).item())
        return torch.tensor(ts, device=device).long()
    except Exception:
        return torch.randint(
            0, scheduler.config.num_train_timesteps,
            (batch_size,), device=device,
        ).long()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SGLDv2 StructureControlNet")

    # Data
    p.add_argument("--data_dir",        type=str, required=True)
    p.add_argument("--no_augment",      action="store_true")
    p.add_argument("--no_structure",    action="store_true",
                   help="Skip StructureExtractor — raw sketch channels only")

    # Model
    p.add_argument("--base_model",      type=str,
                   default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--img_size",        type=int, default=256)
    p.add_argument("--resume",          type=str, default=None)

    # Optimisation
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--grad_accum",      type=int,   default=1,
                   help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-2)
    p.add_argument("--warmup_steps",    type=int,   default=200)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--snr_gamma",       type=float, default=5.0,
                   help="Min-SNR gamma (0 = plain MSE, 5 = recommended)")
    p.add_argument("--clip_weight",     type=float, default=0.05,
                   help="CLIP perceptual loss weight (0 to disable)")
    p.add_argument("--ema_decay",       type=float, default=0.9999,
                   help="EMA decay (0 to disable EMA)")
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--mixed_precision", type=str,   default="fp16",
                   choices=["no", "fp16", "bf16"])

    # Output / logging
    p.add_argument("--output_dir",      type=str, default="./checkpoints")
    p.add_argument("--save_every",      type=int, default=2)
    p.add_argument("--sample_every",    type=int, default=1)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def training_step(
    batch:        dict,
    model:        SGLDv2Model,
    clip_loss_fn: CLIPPerceptualLoss,
    args:         argparse.Namespace,
    device:       str,
) -> torch.Tensor:
    B = batch["photo"].shape[0]

    # 1. Text embeddings (fixed caption — no category)
    encoder_hidden = model.encode_text([FIXED_CAPTION] * B)

    # 2. VAE encode target photo → latents
    latents = model.encode_image(batch["photo"].to(device))

    # 3. Noise + adaptive timestep sampling
    noise     = torch.randn_like(latents)
    timesteps = sample_timesteps(
        B, model.noise_scheduler, batch["structure"], device
    )
    noisy_lat = model.add_noise(latents, noise, timesteps)

    # 4. Forward pass: structure_maps → ControlNet → UNet → noise_pred
    noise_pred = model(
        noisy_latents  = noisy_lat,
        timesteps      = timesteps,
        encoder_hidden = encoder_hidden,
        structure_maps = batch["structure"].to(device),
    )

    # 5. Min-SNR-γ weighted diffusion loss
    loss = model.diffusion_loss(noise_pred, noise, timesteps, args.snr_gamma)

    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite loss: {loss.item()}")

    # 6. Optional CLIP perceptual loss (non-fatal)
    if args.clip_weight > 0 and clip_loss_fn.enabled:
        try:
            with torch.no_grad():
                pred_x0 = model.noise_scheduler.step(
                    noise_pred.float(), timesteps[0], noisy_lat.float()
                ).pred_original_sample
            decoded = model.decode_latents(pred_x0.to(model.dtype))
            clip_l  = clip_loss_fn(decoded.float(),
                                   batch["photo"].float().to(device))
            if torch.isfinite(clip_l):
                loss = loss + args.clip_weight * clip_l
        except Exception as e:
            print(f"  [CLIPLoss] Skipped: {e}")

    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:       SGLDv2Model,
    ema:         Optional[EMAModel],
    path:        str,
    epoch:       int,
    loss:        float,
    accelerator: Accelerator,
) -> bool:
    try:
        os.makedirs(path, exist_ok=True)

        # If EMA is active, save EMA weights as the primary checkpoint
        if ema is not None:
            _cn  = accelerator.unwrap_model(model.control_net)
            ema.copy_to(_cn.parameters())   # temporarily store EMA into model

        model.save(path)

        with open(os.path.join(path, "epoch.txt"), "w") as f:
            f.write(f"epoch={epoch}\nloss={loss:.6f}\n")
        return True
    except Exception as e:
        print(f"  [Save] Checkpoint save failed ({e}) — continuing.")
        return False


def save_samples(batch: dict, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_sample_grid(batch, path)
    except Exception as e:
        print(f"  [Sample] Grid save failed ({e}).")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train() -> None:
    args = parse_args()

    # ── Pre-flight checks ──────────────────────────────────────────────────
    if not os.path.isdir(args.data_dir):
        print(f"[Error] --data_dir not found: '{args.data_dir}'")
        sys.exit(1)
    for sub in ("sketch", "photo"):
        if not os.path.isdir(os.path.join(args.data_dir, sub)):
            print(f"[Error] Missing: {args.data_dir}/{sub}/")
            sys.exit(1)
    if args.batch_size < 1:
        print("[Error] --batch_size must be ≥ 1")
        sys.exit(1)
    if args.grad_accum < 1:
        print("[Error] --grad_accum must be ≥ 1")
        sys.exit(1)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # TF32 on Ampere+ (free speedup, no quality loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # ── Accelerator ────────────────────────────────────────────────────────
    try:
        accelerator = Accelerator(
            mixed_precision          = args.mixed_precision,
            gradient_accumulation_steps = args.grad_accum,
            log_with                 = "tensorboard",
            project_dir              = args.output_dir,
        )
    except Exception as e:
        print(f"[Error] Accelerator init failed: {e}")
        sys.exit(1)

    device = accelerator.device

    if accelerator.is_main_process:
        try:
            log_config(vars(args), args.output_dir)
        except Exception:
            pass
        print(f"\n{'='*62}")
        print("  SGLDv2 Training — StructureControlNet")
        print(f"{'='*62}")
        print(f"  Device          : {device}")
        print(f"  Mixed prec.     : {args.mixed_precision}")
        print(f"  Epochs          : {args.epochs}")
        print(f"  Batch / accum   : {args.batch_size} × {args.grad_accum}"
              f" = {args.batch_size * args.grad_accum} effective")
        print(f"  LR              : {args.lr}")
        print(f"  Min-SNR γ       : {args.snr_gamma}")
        print(f"  EMA decay       : {args.ema_decay}")
        print(f"  CLIP weight     : {args.clip_weight}")
        print(f"  Disk cache      : enabled (structure maps cached as .npz)")
        print(f"{'='*62}\n")

    # ── Structure extractor ────────────────────────────────────────────────
    extractor = None
    if not args.no_structure:
        try:
            extractor = StructureExtractor(device=str(device), use_cache=True)
        except Exception as e:
            print(f"  [Warning] StructureExtractor failed ({e}). Using raw sketch.")

    # ── Dataset ───────────────────────────────────────────────────────────
    print("[1/4] Building dataset …")
    try:
        dataloader = build_dataloader(
            data_dir    = args.data_dir,
            img_size    = args.img_size,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            extractor   = extractor,
            augment     = not args.no_augment,
        )
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(dataloader.dataset) == 0:
        print("[Error] Dataset is empty.")
        sys.exit(1)

    # ── Model ─────────────────────────────────────────────────────────────
    print("[2/4] Loading SGLDv2 model …")
    try:
        model = SGLDv2Model(
            base_model = args.base_model,
            img_size   = args.img_size,
            device     = device,
        )
    except OSError as e:
        print(f"[Error] Could not load '{args.base_model}': {e}")
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        print("[Error] GPU OOM. Try --mixed_precision fp16.")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Model init failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if args.resume:
        try:
            model.load_adapter(args.resume)
            print(f"  Resumed from: {args.resume}")
        except Exception as e:
            print(f"  [Warning] Resume failed ({e}) — starting fresh.")

    if accelerator.is_main_process:
        try:
            n = count_parameters(model.control_net)
            print(f"  Trainable params: {n:,}  (~{n/1e6:.1f}M)")
        except Exception:
            pass

    # ── EMA ───────────────────────────────────────────────────────────────
    ema: Optional[EMAModel] = None
    if args.ema_decay > 0:
        try:
            ema = EMAModel(
                model.control_net.parameters(),
                decay=args.ema_decay,
                model_cls=type(model.control_net),
                model_config=None,
            )
            print(f"  EMA: ENABLED (decay={args.ema_decay})")
        except Exception as e:
            print(f"  [Warning] EMA init failed ({e}) — disabled.")
            ema = None

    clip_loss_fn = CLIPPerceptualLoss(device=str(device))

    # ── Optimiser ─────────────────────────────────────────────────────────
    print("[3/4] Setting up optimiser …")
    try:
        optimiser    = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=args.lr, weight_decay=args.weight_decay,
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimiser,
            num_warmup_steps   = args.warmup_steps,
            num_training_steps = (len(dataloader) // args.grad_accum) * args.epochs,
        )
    except Exception as e:
        print(f"[Error] Optimiser setup failed: {e}")
        sys.exit(1)

    # ── Accelerate ─────────────────────────────────────────────────────────
    try:
        model.control_net, optimiser, dataloader, lr_scheduler = \
            accelerator.prepare(
                model.control_net, optimiser, dataloader, lr_scheduler
            )
    except Exception as e:
        print(f"[Error] accelerator.prepare() failed: {e}")
        sys.exit(1)

    # ── Training loop ──────────────────────────────────────────────────────
    print("[4/4] Starting training …\n")
    global_step          = 0
    best_loss            = float("inf")
    consecutive_failures = 0
    MAX_FAILURES         = 10

    for epoch in range(1, args.epochs + 1):
        meter = AverageMeter()
        bar   = tqdm(
            dataloader,
            desc    = f"Epoch {epoch:03d}/{args.epochs:03d}",
            disable = not accelerator.is_local_main_process,
        )

        for batch_idx, batch in enumerate(bar):
            try:
                with accelerator.accumulate(model.control_net):
                    loss = training_step(
                        batch, model, clip_loss_fn, args, device
                    )
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.trainable_parameters(), args.grad_clip
                        )
                        if ema is not None:
                            ema.step(model.control_net.parameters())

                    optimiser.step()
                    lr_scheduler.step()
                    optimiser.zero_grad(set_to_none=True)  # slightly faster than zero_grad()

                consecutive_failures = 0
                meter.update(loss.item())
                global_step += 1
                bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg":  f"{meter.avg:.4f}",
                    "lr":   f"{lr_scheduler.get_last_lr()[0]:.2e}",
                })

            except ValueError as e:
                # Non-finite loss — skip batch
                print(f"\n  [NaN] Step {batch_idx}: {e}")
                optimiser.zero_grad(set_to_none=True)
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print(f"[Error] {MAX_FAILURES} NaN failures. "
                          "Try: --lr 5e-5 or --mixed_precision no")
                    sys.exit(1)
                continue

            except torch.cuda.OutOfMemoryError:
                print(f"\n  [OOM] Step {batch_idx}. Try --batch_size 2")
                torch.cuda.empty_cache()
                optimiser.zero_grad(set_to_none=True)
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print("[Error] Repeated OOM — aborting.")
                    sys.exit(1)
                continue

            except RuntimeError as e:
                print(f"\n  [RuntimeError] Step {batch_idx}: {e}")
                optimiser.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    traceback.print_exc()
                    sys.exit(1)
                continue

            except KeyboardInterrupt:
                print("\n  Interrupted — saving emergency checkpoint …")
                save_checkpoint(
                    model, ema,
                    os.path.join(args.output_dir, "interrupted"),
                    epoch, meter.avg if meter.count > 0 else float("inf"),
                    accelerator,
                )
                print("  Saved. Resume with --resume checkpoints/interrupted")
                sys.exit(0)

            except Exception as e:
                print(f"\n  [Error] Unexpected at step {batch_idx}: {e}")
                traceback.print_exc()
                optimiser.zero_grad(set_to_none=True)
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    sys.exit(1)
                continue

        # ── End of epoch ───────────────────────────────────────────────────
        if accelerator.is_main_process:
            avg     = meter.avg if meter.count > 0 else float("inf")
            skipped = len(dataloader) - meter.count
            msg     = f"  Epoch {epoch:03d} | loss={avg:.4f}"
            if skipped > 0:
                msg += f" | skipped={skipped}"
            print(msg)

            try:
                accelerator.log(
                    {"loss": avg, "lr": lr_scheduler.get_last_lr()[0]},
                    step=global_step,
                )
            except Exception:
                pass

            if epoch % args.save_every == 0:
                ckpt  = os.path.join(args.output_dir, "latest")
                saved = save_checkpoint(model, ema, ckpt, epoch, avg, accelerator)
                if saved:
                    print(f"  ✓ Latest → {ckpt}")

            if avg < best_loss:
                best_loss = avg
                saved = save_checkpoint(
                    model, ema,
                    os.path.join(args.output_dir, "best"),
                    epoch, avg, accelerator,
                )
                if saved:
                    print(f"  ★ Best (loss={best_loss:.4f}) → checkpoints/best")

            if epoch % args.sample_every == 0:
                save_samples(
                    batch,
                    os.path.join(args.output_dir, "samples", f"ep{epoch:03d}.png"),
                )

    # ── Final save ─────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        saved = save_checkpoint(
            model, ema,
            os.path.join(args.output_dir, "final"),
            args.epochs, best_loss, accelerator,
        )
        if saved:
            print(f"\n  Training complete.")
            print(f"  Best loss : {best_loss:.4f}")
            print(f"  Output    : {args.output_dir}")


if __name__ == "__main__":
    from typing import Optional   # needed for Python < 3.10
    try:
        train()
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        traceback.print_exc()
        sys.exit(1)
