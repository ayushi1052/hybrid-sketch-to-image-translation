"""
train.py
=========
Training script for the modified d-Sketch pipeline.

What trains
────────────
  LCTN                 — translates sketch latent → image latent
  ColoredEdgeGenerator — predicts colored edge map from sketch
  ImageConditionProj   — projects CLIP vision features → SD conditioning

What is frozen (never updated)
───────────────────────────────
  VAE, SD UNet, CLIP Vision Encoder

Loss
─────
  L_total = L_latent + 0.1 × L_perceptual + 0.5 × L_edge_color

Usage
─────
  python train.py --data_dir ./dataset --epochs 30
  python train.py --data_dir ./dataset --resume ./checkpoints/latest
  python train.py --data_dir ./dataset --batch_size 2 --img_size 256
"""

from __future__ import annotations

import os
import sys
import argparse
import traceback
import torch
from tqdm import tqdm
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision.utils import make_grid, save_image

from models          import SketchToImagePipeline
from dataset_loader  import build_dataloader
from utils           import set_seed, AverageMeter, log_config, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train modified d-Sketch pipeline"
    )
    # Data
    p.add_argument("--data_dir",        type=str,   required=True)
    p.add_argument("--img_size",        type=int,   default=256)
    p.add_argument("--no_augment",      action="store_true")

    # Model
    p.add_argument("--sd_model",        type=str,
                   default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--clip_model",      type=str,
                   default="openai/clip-vit-large-patch14")
    p.add_argument("--k_ratio",         type=float, default=0.8,
                   help="Noise ratio k/T (0.7-0.9). Higher=more creative, "
                        "Lower=more faithful to sketch structure")
    p.add_argument("--resume",          type=str,   default=None)

    # Training
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--grad_accum",      type=int,   default=1)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-2)
    p.add_argument("--warmup_steps",    type=int,   default=500)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--mixed_precision", type=str,   default="fp16",
                   choices=["no", "fp16", "bf16"])

    # Output
    p.add_argument("--output_dir",      type=str,   default="./checkpoints")
    p.add_argument("--save_every",      type=int,   default=5)
    p.add_argument("--sample_every",    type=int,   default=2)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    pipeline:    SketchToImagePipeline,
    path:        str,
    epoch:       int,
    loss:        float,
    accelerator: Accelerator,
) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(pipeline)
        unwrapped.save(path)
        with open(os.path.join(path, "epoch.txt"), "w") as f:
            f.write(f"epoch={epoch}\nloss={loss:.6f}\n")
        return True
    except Exception as e:
        print(f"  [Save] Failed ({e})")
        return False


def save_sample_grid(
    batch:     dict,
    pipeline:  SketchToImagePipeline,
    path:      str,
) -> None:
    """Save grid: Sketch | Target Colored Map | Predicted Colored Map"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sk   = batch["sketch"][:4].cpu()          * 0.5 + 0.5
        tgt  = batch["target_colored"][:4].cpu()  * 0.5 + 0.5

        with torch.no_grad():
            pred = pipeline.edge_gen(
                batch["sketch"][:4].to(pipeline.device)
            ).float().cpu() * 0.5 + 0.5

        rows  = []
        for i in range(min(4, sk.shape[0])):
            rows.extend([sk[i], tgt[i], pred[i]])

        grid  = make_grid(torch.stack(rows), nrow=3, padding=4, pad_value=1.0)
        save_image(grid, path)
        print(f"  [Sample] Grid saved → {path}")
    except Exception as e:
        print(f"  [Sample] Grid save failed ({e})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train() -> None:
    args = parse_args()

    # ── Pre-flight ────────────────────────────────────────────────────────
    if not os.path.isdir(args.data_dir):
        print(f"[Error] --data_dir not found: '{args.data_dir}'")
        sys.exit(1)
    for sub in ("sketch", "photo"):
        if not os.path.isdir(os.path.join(args.data_dir, sub)):
            print(f"[Error] Missing: {args.data_dir}/{sub}/")
            sys.exit(1)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # TF32 free speedup on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # ── Accelerator ────────────────────────────────────────────────────────
    try:
        accelerator = Accelerator(
            mixed_precision             = args.mixed_precision,
            gradient_accumulation_steps = args.grad_accum,
            log_with                    = "tensorboard",
            project_dir                 = args.output_dir,
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
        print(f"\n{'='*60}")
        print("  Modified d-Sketch Training")
        print(f"{'='*60}")
        print(f"  Device       : {device}")
        print(f"  Precision    : {args.mixed_precision}")
        print(f"  Epochs       : {args.epochs}")
        print(f"  Batch/accum  : {args.batch_size} × {args.grad_accum}")
        print(f"  k_ratio      : {args.k_ratio}")
        print(f"  LR           : {args.lr}")
        print(f"{'='*60}")
        print()
        print("  What trains:")
        print("    LCTN                 — sketch latent → image latent")
        print("    ColoredEdgeGenerator — sketch → colored edge map")
        print("    ImageConditionProj   — CLIP features → SD conditioning")
        print()
        print("  Frozen: VAE, SD UNet, CLIP Vision")
        print(f"{'='*60}\n")

    # ── Dataset ───────────────────────────────────────────────────────────
    print("[1/4] Building dataset …")
    try:
        dataloader = build_dataloader(
            data_dir    = args.data_dir,
            img_size    = args.img_size,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            augment     = not args.no_augment,
        )
    except Exception as e:
        print(f"[Error] Dataset: {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(dataloader.dataset) == 0:
        print("[Error] Dataset is empty.")
        sys.exit(1)

    # ── Pipeline ──────────────────────────────────────────────────────────
    print("[2/4] Loading pipeline …")
    try:
        pipeline = SketchToImagePipeline(
            sd_model   = args.sd_model,
            clip_model = args.clip_model,
            img_size   = args.img_size,
            device     = device,
            k_ratio    = args.k_ratio,
        )
    except OSError as e:
        print(f"[Error] Model load failed: {e}")
        print("        Check internet connection or set HF_TOKEN.")
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        print("[Error] GPU OOM. Try --mixed_precision fp16 or --img_size 256")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Pipeline init failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if args.resume:
        try:
            pipeline.load(args.resume)
            print(f"  Resumed from: {args.resume}")
        except Exception as e:
            print(f"  [Warning] Resume failed ({e}) — starting fresh.")

    if accelerator.is_main_process:
        n = count_parameters(pipeline.lctn) \
          + count_parameters(pipeline.edge_gen) \
          + count_parameters(pipeline.cond_proj)
        print(f"  Trainable params: {n:,}  (~{n/1e6:.1f}M)\n")

    # ── Optimiser ─────────────────────────────────────────────────────────
    print("[3/4] Setting up optimiser …")
    try:
        optimiser    = torch.optim.AdamW(
            pipeline.trainable_parameters(),
            lr           = args.lr,
            weight_decay = args.weight_decay,
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
        pipeline, optimiser, dataloader, lr_scheduler = accelerator.prepare(
            pipeline, optimiser, dataloader, lr_scheduler
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
        meter_total = AverageMeter()
        meter_lat   = AverageMeter()
        meter_edge  = AverageMeter()

        bar = tqdm(
            dataloader,
            desc    = f"Epoch {epoch:03d}/{args.epochs:03d}",
            disable = not accelerator.is_local_main_process,
        )

        for batch_idx, batch in enumerate(bar):
            try:
                with accelerator.accumulate(pipeline):
                    outputs = pipeline(
                        sketch          = batch["sketch"],
                        real_image      = batch["real_image"],
                        target_colored  = batch["target_colored"],
                    )

                    loss = outputs["loss_total"]

                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite loss: {loss.item()}")

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            pipeline.trainable_parameters(), args.grad_clip
                        )

                    optimiser.step()
                    lr_scheduler.step()
                    optimiser.zero_grad(set_to_none=True)

                consecutive_failures = 0
                meter_total.update(loss.item())
                meter_lat.update(outputs["loss_latent"].item())
                meter_edge.update(outputs["loss_edge_color"].item())
                global_step += 1

                bar.set_postfix({
                    "loss":  f"{loss.item():.4f}",
                    "lat":   f"{outputs['loss_latent'].item():.4f}",
                    "edge":  f"{outputs['loss_edge_color'].item():.4f}",
                    "lr":    f"{lr_scheduler.get_last_lr()[0]:.2e}",
                })

            except ValueError as e:
                print(f"\n  [NaN] Step {batch_idx}: {e}")
                optimiser.zero_grad(set_to_none=True)
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print("[Error] Too many NaN losses. Try --lr 5e-5")
                    sys.exit(1)
                continue

            except torch.cuda.OutOfMemoryError:
                print(f"\n  [OOM] Step {batch_idx}. Try --batch_size 4")
                torch.cuda.empty_cache()
                optimiser.zero_grad(set_to_none=True)
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    sys.exit(1)
                continue

            except KeyboardInterrupt:
                print("\n  Interrupted — saving emergency checkpoint …")
                save_checkpoint(
                    pipeline, os.path.join(args.output_dir, "interrupted"),
                    epoch, meter_total.avg, accelerator,
                )
                sys.exit(0)

            except Exception as e:
                print(f"\n  [Error] Step {batch_idx}: {e}")
                traceback.print_exc()
                optimiser.zero_grad(set_to_none=True)
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    sys.exit(1)
                continue

        # ── End of epoch ───────────────────────────────────────────────────
        if accelerator.is_main_process:
            avg = meter_total.avg if meter_total.count > 0 else float("inf")
            print(
                f"  Epoch {epoch:03d} | "
                f"total={avg:.4f} | "
                f"latent={meter_lat.avg:.4f} | "
                f"edge={meter_edge.avg:.4f}"
            )

            try:
                accelerator.log(
                    {
                        "loss/total":      avg,
                        "loss/latent":     meter_lat.avg,
                        "loss/edge_color": meter_edge.avg,
                        "lr":              lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
            except Exception:
                pass

            # Save checkpoint
            if epoch % args.save_every == 0:
                ckpt  = os.path.join(args.output_dir, "latest")
                saved = save_checkpoint(pipeline, ckpt, epoch, avg, accelerator)
                if saved:
                    print(f"  ✓ Latest → {ckpt}")

            # Best model
            if avg < best_loss:
                best_loss = avg
                saved = save_checkpoint(
                    pipeline,
                    os.path.join(args.output_dir, "best"),
                    epoch, avg, accelerator,
                )
                if saved:
                    print(f"  ★ Best (loss={best_loss:.4f}) → checkpoints/best")

            # Sample grid
            if epoch % args.sample_every == 0:
                save_sample_grid(
                    batch,
                    accelerator.unwrap_model(pipeline),
                    os.path.join(args.output_dir, "samples", f"ep{epoch:03d}.png"),
                )

    # ── Final save ─────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        saved = save_checkpoint(
            pipeline,
            os.path.join(args.output_dir, "final"),
            args.epochs, best_loss, accelerator,
        )
        if saved:
            print(f"\n  Training complete.")
            print(f"  Best loss : {best_loss:.4f}")
            print(f"  Output    : {args.output_dir}")


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