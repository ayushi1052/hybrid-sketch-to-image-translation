"""
train.py
=========
SGLDv2 Training Script — with full exception handling

Usage:
  python train.py --data_dir ./dataset --epochs 20
  python train.py --data_dir ./dataset --epochs 3 --no_structure
  python train.py --data_dir ./dataset --resume ./checkpoints/latest
"""

import os
import sys
import argparse
import traceback
import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup

from models         import SGLDv2Model, StructureExtractor
from dataset_loader import build_dataloader
from utils          import (
    set_seed, AverageMeter, log_config,
    save_sample_grid, count_parameters,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP Perceptual Loss
# ─────────────────────────────────────────────────────────────────────────────

class CLIPPerceptualLoss(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        try:
            import clip
            self.model, _ = clip.load("ViT-B/32", device=device)
            self.model.requires_grad_(False)
            self.enabled = True
            print("  [CLIPLoss] CLIP perceptual loss: ENABLED")
        except ImportError:
            self.enabled = False
            print("  [CLIPLoss] clip not installed — MSE only.\n"
                  "            Run: pip install git+https://github.com/openai/CLIP.git")
        except Exception as e:
            self.enabled = False
            print(f"  [CLIPLoss] Failed to load CLIP ({e}) — MSE only.")

    def forward(self, gen: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.tensor(0.0, device=gen.device)
        try:
            g = F.interpolate((gen * 0.5 + 0.5).clamp(0, 1), (224, 224))
            t = F.interpolate((tgt * 0.5 + 0.5).clamp(0, 1), (224, 224))
            return 1 - F.cosine_similarity(
                self.model.encode_image(g),
                self.model.encode_image(t)
            ).mean()
        except Exception as e:
            print(f"  [CLIPLoss] Forward failed ({e}) — skipping CLIP loss this step.")
            return torch.tensor(0.0, device=gen.device)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive timestep sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_timesteps(batch_size, scheduler, structure_maps, device):
    """
    Adaptive sampling — falls back to uniform if anything goes wrong.
    """
    try:
        T            = scheduler.config.num_train_timesteps
        edge_density = structure_maps[:, 0].mean(dim=[1, 2]).cpu()
        alpha        = 1.0 + (1.0 - edge_density) * 3.0
        timesteps    = []
        for a in alpha:
            probs = torch.arange(1, T + 1, dtype=torch.float) ** float(a)
            probs = probs / probs.sum()
            timesteps.append(torch.multinomial(probs, 1).item())
        return torch.tensor(timesteps, device=device).long()
    except Exception as e:
        print(f"  [Timestep] Adaptive sampling failed ({e}) — using uniform fallback.")
        return torch.randint(
            0, scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",        type=str, required=True)
    p.add_argument("--output_dir",      type=str, default="./checkpoints")
    p.add_argument("--resume",          type=str, default=None)
    p.add_argument("--base_model",      type=str,
                   default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--num_tokens",      type=int,   default=16)
    p.add_argument("--img_size",        type=int,   default=256)
    p.add_argument("--categories",      type=str, nargs="+", default=None)
    p.add_argument("--no_augment",      action="store_true")
    p.add_argument("--no_structure",    action="store_true")
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-2)
    p.add_argument("--warmup_steps",    type=int,   default=200)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--clip_weight",     type=float, default=0.1)
    p.add_argument("--num_workers",     type=int,   default=2)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--mixed_precision", type=str,   default="fp16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--save_every",      type=int,   default=2)
    p.add_argument("--sample_every",    type=int,   default=1)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def training_step(batch, model, clip_loss_fn, args, device):
    # 1. Text embeddings
    try:
        encoder_hidden = model.encode_text(list(batch["caption"]))
    except Exception as e:
        raise RuntimeError(f"CLIP text encoding failed: {e}") from e

    # 2. VAE encode photo → latent
    try:
        latents = model.encode_image(batch["photo"].to(device))
    except Exception as e:
        raise RuntimeError(f"VAE encoding failed: {e}") from e

    # 3. Noise + timesteps
    try:
        noise     = torch.randn_like(latents)
        timesteps = sample_timesteps(
            latents.shape[0], model.noise_scheduler,
            batch["structure"], device
        )
        noisy_lat = model.add_noise(latents, noise, timesteps)
    except Exception as e:
        raise RuntimeError(f"Noise/timestep setup failed: {e}") from e

    # 4. Forward pass: structure_maps → sketch_tokens → cat(text+sketch) → UNet
    try:
        noise_pred = model(
            noisy_latents  = noisy_lat,
            timesteps      = timesteps,
            encoder_hidden = encoder_hidden,
            structure_maps = batch["structure"].to(device),
        )
    except Exception as e:
        raise RuntimeError(f"Model forward pass failed: {e}") from e

    # 5. Diffusion MSE loss
    try:
        loss = F.mse_loss(noise_pred.float(), noise.float())
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite MSE loss: {loss.item()}")
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"MSE loss failed: {e}") from e

    # 6. Optional CLIP perceptual loss (non-fatal)
    if args.clip_weight > 0 and clip_loss_fn.enabled:
        try:
            with torch.no_grad():
                pred_x0 = model.noise_scheduler.step(
                    noise_pred.float(), timesteps[0], noisy_lat.float()
                ).pred_original_sample
            decoded = model.decode_latents(pred_x0.to(model.dtype))
            clip_l  = clip_loss_fn(
                decoded.float(), batch["photo"].float().to(device)
            )
            if torch.isfinite(clip_l):
                loss = loss + args.clip_weight * clip_l
            else:
                print("  [CLIPLoss] Non-finite value — skipping this step.")
        except Exception as e:
            print(f"  [CLIPLoss] Skipped this step ({e}).")

    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, path, epoch, loss, accelerator):
    """Save checkpoint — non-fatal, logs warning on failure."""
    try:
        os.makedirs(path, exist_ok=True)
        accelerator.unwrap_model(model.sketch_adapter)
        model.save(path)
        with open(f"{path}/epoch.txt", "w") as f:
            f.write(f"epoch={epoch}\nloss={loss:.6f}\n")
        return True
    except PermissionError:
        print(f"  [Save] Permission denied writing to {path}.")
        return False
    except OSError as e:
        print(f"  [Save] OS error while saving ({e}) — continuing training.")
        return False
    except Exception as e:
        print(f"  [Save] Checkpoint save failed ({e}) — continuing training.")
        return False


def save_samples(batch, path):
    """Save visual grid — non-fatal."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_sample_grid(batch, path)
    except Exception as e:
        print(f"  [Sample] Sample grid save failed ({e}) — skipping.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train():
    args = parse_args()

    # ── Pre-flight checks ──────────────────────────────────────────────────
    if not os.path.isdir(args.data_dir):
        print(f"[Error] --data_dir not found: '{args.data_dir}'")
        sys.exit(1)

    for sub in ["sketch", "photo"]:
        path = os.path.join(args.data_dir, sub)
        if not os.path.isdir(path):
            print(f"[Error] Missing required subfolder: {path}")
            print(f"        Expected structure:")
            print(f"          {args.data_dir}/sketch/<category>/img.jpg")
            print(f"          {args.data_dir}/photo/<category>/img.jpg")
            sys.exit(1)

    if args.resume and not os.path.isdir(args.resume):
        print(f"[Error] --resume path not found: '{args.resume}'")
        sys.exit(1)

    if args.batch_size < 1:
        print("[Error] --batch_size must be >= 1")
        sys.exit(1)

    if args.lr <= 0:
        print("[Error] --lr must be > 0")
        sys.exit(1)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Accelerator ────────────────────────────────────────────────────────
    try:
        accelerator = Accelerator(
            mixed_precision = args.mixed_precision,
            log_with        = "tensorboard",
            project_dir     = args.output_dir,
        )
    except Exception as e:
        print(f"[Error] Accelerator init failed: {e}")
        print("        Try: --mixed_precision no")
        sys.exit(1)

    device = accelerator.device

    if accelerator.is_main_process:
        try:
            log_config(vars(args), args.output_dir)
        except Exception as e:
            print(f"  [Warning] Could not save config.json: {e}")

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
    extractor = None
    if not args.no_structure:
        try:
            extractor = StructureExtractor(device=device)
        except Exception as e:
            print(f"  [Warning] StructureExtractor failed ({e}).")
            print(f"            Falling back to raw sketch channels.")

    # ── Dataset ───────────────────────────────────────────────────────────
    print("[1/4] Building dataset …")
    try:
        dataloader = build_dataloader(
            data_dir    = args.data_dir,
            img_size    = args.img_size,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            extractor   = extractor,
            categories  = args.categories,
            augment     = not args.no_augment,
        )
    except FileNotFoundError as e:
        print(f"[Error] Dataset folder not found: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[Error] Dataset error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Unexpected dataset error: {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(dataloader.dataset) == 0:
        print("[Error] Dataset is empty — no sketch/photo pairs found.")
        print(f"        Check: {args.data_dir}/sketch/ and {args.data_dir}/photo/")
        sys.exit(1)

    # ── Model ─────────────────────────────────────────────────────────────
    print("[2/4] Loading SGLDv2 model …")
    try:
        model = SGLDv2Model(
            base_model = args.base_model,
            img_size   = args.img_size,
            num_tokens = args.num_tokens,
            device     = device,
        )
    except OSError as e:
        print(f"[Error] Could not download/load '{args.base_model}': {e}")
        print("        Check internet connection or set HF_TOKEN env variable.")
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        print("[Error] GPU out of memory loading model.")
        print("        Try: --mixed_precision fp16  or use a smaller base model.")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Model init failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if args.resume:
        try:
            model.load_adapter(args.resume)
            print(f"  Resumed from: {args.resume}")
        except FileNotFoundError:
            print(f"  [Warning] Resume checkpoint not found — starting from scratch.")
        except Exception as e:
            print(f"  [Warning] Could not load resume checkpoint ({e}) — starting from scratch.")

    if accelerator.is_main_process:
        try:
            n = count_parameters(model.sketch_adapter)
            print(f"  Trainable params: {n:,}  (~{n/1e6:.1f}M)")
        except Exception:
            pass

    clip_loss_fn = CLIPPerceptualLoss(device=device)

    # ── Optimiser ─────────────────────────────────────────────────────────
    print("[3/4] Setting up optimiser …")
    try:
        optimiser = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=args.lr, weight_decay=args.weight_decay,
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimiser,
            num_warmup_steps   = args.warmup_steps,
            num_training_steps = len(dataloader) * args.epochs,
        )
    except Exception as e:
        print(f"[Error] Optimiser setup failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Accelerate ─────────────────────────────────────────────────────────
    try:
        model.sketch_adapter, optimiser, dataloader, lr_scheduler = \
            accelerator.prepare(
                model.sketch_adapter, optimiser, dataloader, lr_scheduler
            )
    except Exception as e:
        print(f"[Error] accelerator.prepare() failed: {e}")
        traceback.print_exc()
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
                with accelerator.accumulate(model.sketch_adapter):
                    loss = training_step(
                        batch, model, clip_loss_fn, args, device
                    )

                    # NaN / Inf guard
                    if not torch.isfinite(loss):
                        print(f"\n  [Warning] Non-finite loss at epoch {epoch} "
                              f"step {batch_idx} — skipping batch.")
                        consecutive_failures += 1
                        if consecutive_failures >= MAX_FAILURES:
                            print(f"[Error] {MAX_FAILURES} consecutive failures.")
                            print("        Suggestions:")
                            print("          • Lower --lr (try 1e-5)")
                            print("          • Use --mixed_precision no")
                            print("          • Add --no_structure")
                            sys.exit(1)
                        optimiser.zero_grad()
                        continue

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.trainable_parameters(), args.grad_clip
                        )

                    optimiser.step()
                    lr_scheduler.step()
                    optimiser.zero_grad()

                consecutive_failures = 0
                meter.update(loss.item())
                global_step += 1
                bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg":  f"{meter.avg:.4f}",
                    "lr":   f"{lr_scheduler.get_last_lr()[0]:.2e}",
                })

            except torch.cuda.OutOfMemoryError:
                print(f"\n  [OOM] GPU out of memory at step {batch_idx}.")
                print("        Try: lower --batch_size (e.g. --batch_size 2)")
                torch.cuda.empty_cache()
                optimiser.zero_grad()
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print("[Error] Repeated OOM — aborting.")
                    sys.exit(1)
                continue

            except RuntimeError as e:
                print(f"\n  [RuntimeError] Step {batch_idx}: {e}")
                optimiser.zero_grad()
                torch.cuda.empty_cache()
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print(f"[Error] {MAX_FAILURES} consecutive failures — aborting.")
                    traceback.print_exc()
                    sys.exit(1)
                continue

            except KeyboardInterrupt:
                print("\n\n  Interrupted by user — saving emergency checkpoint …")
                save_checkpoint(
                    model,
                    os.path.join(args.output_dir, "interrupted"),
                    epoch, meter.avg if meter.count > 0 else float("inf"),
                    accelerator
                )
                print("  Saved → checkpoints/interrupted  (resume with --resume)")
                sys.exit(0)

            except Exception as e:
                print(f"\n  [Error] Unexpected error at step {batch_idx}: {e}")
                traceback.print_exc()
                optimiser.zero_grad()
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print("[Error] Too many failures — aborting.")
                    sys.exit(1)
                continue

        # ── End of epoch ───────────────────────────────────────────────────
        if accelerator.is_main_process:
            avg = meter.avg if meter.count > 0 else float("inf")
            skipped = len(dataloader) - meter.count
            msg = f"  Epoch {epoch:03d} | loss={avg:.4f}"
            if skipped > 0:
                msg += f" | skipped={skipped} batches"
            print(msg)

            try:
                accelerator.log(
                    {"loss": avg, "lr": lr_scheduler.get_last_lr()[0]},
                    step=global_step
                )
            except Exception:
                pass

            if epoch % args.save_every == 0:
                ckpt  = os.path.join(args.output_dir, "latest")
                saved = save_checkpoint(model, ckpt, epoch, avg, accelerator)
                if saved:
                    print(f"  ✓ Latest updated (epoch {epoch}) → {ckpt}")

            if avg < best_loss:
                best_loss = avg
                saved = save_checkpoint(
                    model,
                    os.path.join(args.output_dir, "best"),
                    epoch, avg, accelerator
                )
                if saved:
                    print(f"  ★ Best (loss={best_loss:.4f}) → checkpoints/best")

            if epoch % args.sample_every == 0:
                save_samples(
                    batch,
                    os.path.join(args.output_dir, "samples", f"ep{epoch:03d}.png")
                )

    # ── Final save ─────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        saved = save_checkpoint(
            model,
            os.path.join(args.output_dir, "final"),
            args.epochs, best_loss, accelerator
        )
        if saved:
            print(f"\n  Training complete.")
            print(f"  Best loss : {best_loss:.4f}")
            print(f"  Output    : {args.output_dir}")
        else:
            print(f"\n  [Warning] Final save failed — check {args.output_dir}")


if __name__ == "__main__":
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