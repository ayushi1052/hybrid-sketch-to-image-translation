"""
inference.py
=============
SGLDv2 Inference — with full exception handling

Usage:
  python inference.py --sketch ./cat.png --output ./result.png --category cat
  python inference.py --sketch_dir ./dataset/sketch/airplane --output_dir ./results
  python inference.py --demo --adapter ./checkpoints/final
"""

import os
import sys
import argparse
import traceback
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from models         import SGLDv2InferencePipeline, StructureExtractor
from dataset_loader import build_caption
from utils          import set_seed, save_comparison


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",         type=str, default="./checkpoints/final")
    p.add_argument("--base_model",      type=str,
                   default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--num_tokens",      type=int,   default=16)
    p.add_argument("--img_size",        type=int,   default=256)
    p.add_argument("--sketch",          type=str,   default=None)
    p.add_argument("--output",          type=str,   default="./output.png")
    p.add_argument("--sketch_dir",      type=str,   default=None)
    p.add_argument("--output_dir",      type=str,   default="./results")
    p.add_argument("--demo",            action="store_true")
    p.add_argument("--prompt",          type=str,
                   default="a realistic photo, highly detailed, sharp focus")
    p.add_argument("--neg_prompt",      type=str,
                   default="low quality, blurry, sketch, cartoon, line drawing")
    p.add_argument("--steps",           type=int,   default=30)
    p.add_argument("--guidance",        type=float, default=7.5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--category",        type=str,   default=None)
    p.add_argument("--no_structure",    action="store_true")
    p.add_argument("--save_comparison", action="store_true")
    p.add_argument("--device",          type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline loader
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline(args):
    """Load inference pipeline with clear error messages."""
    if not os.path.isdir(args.adapter):
        print(f"[Error] Adapter checkpoint not found: '{args.adapter}'")
        print("        Train first:  python train.py --data_dir ./dataset")
        sys.exit(1)

    adapter_file = os.path.join(args.adapter, "sketch_adapter.pt")
    if not os.path.isfile(adapter_file):
        print(f"[Error] No 'sketch_adapter.pt' found in: {args.adapter}")
        print("        Make sure you point to a valid checkpoint folder.")
        sys.exit(1)

    try:
        pipeline = SGLDv2InferencePipeline(
            adapter_path = args.adapter,
            base_model   = args.base_model,
            device       = args.device,
            num_tokens   = args.num_tokens,
            img_size     = args.img_size,
        )
        return pipeline
    except OSError as e:
        print(f"[Error] Could not load base model '{args.base_model}': {e}")
        print("        Check your internet connection.")
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        print("[Error] GPU out of memory loading model.")
        print("        Try: --device cpu")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Pipeline load failed: {e}")
        traceback.print_exc()
        sys.exit(1)


def load_extractor(args):
    """Load structure extractor — returns None on failure (non-fatal)."""
    if args.no_structure:
        return None
    try:
        return StructureExtractor(device=args.device)
    except Exception as e:
        print(f"  [Warning] StructureExtractor failed ({e}).")
        print("            Using raw sketch as fallback.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Single image generation
# ─────────────────────────────────────────────────────────────────────────────

def load_sketch(path):
    """Load and validate a sketch image."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sketch file not found: '{path}'")

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if Path(path).suffix.lower() not in valid_exts:
        raise ValueError(
            f"Unsupported file type: '{Path(path).suffix}'. "
            f"Supported: {valid_exts}"
        )

    try:
        img = Image.open(path).convert("RGB")
        if img.size[0] < 16 or img.size[1] < 16:
            raise ValueError(f"Image too small: {img.size}. Minimum is 16×16.")
        return img
    except (OSError, IOError) as e:
        raise IOError(f"Could not open image '{path}': {e}") from e


def build_structure(sketch_pil, extractor, args):
    """Build structure tensor — falls back to raw sketch on failure."""
    size  = args.img_size
    sk_np = np.array(sketch_pil.convert("RGB").resize((size, size)))

    if extractor is not None:
        try:
            structure = extractor.extract(sk_np, (size, size))
            ctrl_pil  = extractor.build_control_pil(sk_np, size, "combined")
            return structure, ctrl_pil
        except Exception as e:
            print(f"  [Warning] Structure extraction failed ({e}) — using raw sketch.")

    # Fallback
    from torchvision import transforms
    t         = transforms.ToTensor()(Image.fromarray(sk_np))
    structure = torch.cat([t, torch.zeros(2, size, size)], dim=0)
    ctrl_pil  = Image.fromarray(sk_np)
    return structure, ctrl_pil


def generate(sketch_pil, pipeline, extractor, args, prompt=None):
    """
    Generate one image from a sketch.
    Returns (result_pil, ctrl_pil) or raises on failure.
    """
    prompt    = prompt or args.prompt
    structure, ctrl_pil = build_structure(sketch_pil, extractor, args)

    try:
        result = pipeline.generate(
            structure_maps  = structure,
            prompt          = prompt,
            negative_prompt = args.neg_prompt,
            guidance_scale  = args.guidance,
            num_steps       = args.steps,
            seed            = args.seed,
            width           = args.img_size,
            height          = args.img_size,
        )
        return result, ctrl_pil
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            "GPU out of memory during generation.\n"
            "  Try: --steps 20  or  --img_size 256  or  --device cpu"
        )
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}") from e


def save_result(result, path):
    """Save generated image — raises with clear message on failure."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        result.save(path)
    except PermissionError:
        raise PermissionError(f"Permission denied saving to: '{path}'")
    except OSError as e:
        raise OSError(f"Could not save image to '{path}': {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Gradio Demo
# ─────────────────────────────────────────────────────────────────────────────

def launch_demo(pipeline, extractor, args):
    try:
        import gradio as gr
    except ImportError:
        print("[Error] Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    def on_generate(sketch_np, prompt, neg_prompt, guidance, steps, seed, category):
        if sketch_np is None:
            return None, None, "⚠️ Please upload a sketch."
        try:
            pil    = Image.fromarray(sketch_np).convert("RGB")
            if pil.size[0] < 16 or pil.size[1] < 16:
                return None, None, "⚠️ Image too small (min 16×16)."
            auto_p = build_caption(category.strip()) if category and category.strip() \
                     else prompt
            result, ctrl = generate(pil, pipeline, extractor, args, prompt=auto_p)
            return ctrl, result, "✓ Done"
        except RuntimeError as e:
            return None, None, f"❌ {e}"
        except Exception as e:
            traceback.print_exc()
            return None, None, f"❌ Unexpected error: {e}"

    def on_preview(sketch_np, mode):
        if sketch_np is None:
            return None
        if extractor is None:
            return Image.fromarray(sketch_np)
        try:
            sk = np.array(
                Image.fromarray(sketch_np).convert("RGB").resize(
                    (args.img_size, args.img_size)
                )
            )
            return extractor.build_control_pil(sk, args.img_size, mode)
        except Exception as e:
            print(f"  [Demo] Structure preview failed: {e}")
            return None

    with gr.Blocks(title="SGLDv2", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## 🎨 SGLDv2 — Structure-Guided Latent Diffusion v2\n"
            "Sketch → Realistic Photo"
        )
        with gr.Row():
            with gr.Column():
                sketch_in = gr.Image(label="✏️ Sketch", type="numpy", height=350)
                category  = gr.Textbox(
                    label="Category (auto-prompt)",
                    placeholder="e.g. airplane, cat, apple"
                )
                with gr.Row():
                    mode       = gr.Dropdown(
                        ["edge", "depth", "combined"], value="combined",
                        label="Preview Mode"
                    )
                    preview_btn= gr.Button("👁 Preview Structure")
            with gr.Column():
                ctrl_out   = gr.Image(label="📐 Structure Map", type="pil", height=170)
                result_out = gr.Image(label="🖼 Generated", type="pil", height=350)

        with gr.Row():
            prompt   = gr.Textbox(
                label="Prompt",
                value="a realistic photo, highly detailed, sharp focus",
                lines=2, scale=3
            )
            neg_p    = gr.Textbox(
                label="Negative Prompt",
                value="low quality, blurry, sketch, cartoon",
                lines=2, scale=3
            )
        with gr.Row():
            guidance = gr.Slider(1.0, 20.0, 7.5, step=0.5, label="Guidance Scale")
            steps    = gr.Slider(10, 50, 30, step=5,       label="Steps")
            seed     = gr.Number(42, label="Seed", precision=0)

        status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            gr.Button("🚀 Generate", variant="primary").click(
                on_generate,
                [sketch_in, prompt, neg_p, guidance, steps, seed, category],
                [ctrl_out, result_out, status]
            )
            gr.Button("🗑 Clear", variant="stop").click(
                lambda: (None, None, None, ""),
                outputs=[sketch_in, ctrl_out, result_out, status]
            )
        preview_btn.click(on_preview, [sketch_in, mode], ctrl_out)

    try:
        demo.launch(share=True, server_port=7860)
    except OSError as e:
        print(f"  [Demo] Port 7860 busy ({e}). Trying random port …")
        demo.launch(share=True)
    except Exception as e:
        print(f"[Error] Demo launch failed: {e}")
        traceback.print_exc()
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # Validate args
    if args.steps < 1:
        print("[Error] --steps must be >= 1")
        sys.exit(1)
    if not (1.0 <= args.guidance <= 30.0):
        print("[Warning] --guidance is outside normal range [1.0, 30.0]. Proceeding anyway.")

    # Category → auto-prompt
    if args.category:
        try:
            args.prompt = build_caption(args.category)
            print(f"  Auto-prompt: {args.prompt}")
        except Exception as e:
            print(f"  [Warning] Could not build caption for '{args.category}': {e}")

    pipeline  = load_pipeline(args)
    extractor = load_extractor(args)

    # ── Demo mode ──────────────────────────────────────────────────────────
    if args.demo:
        launch_demo(pipeline, extractor, args)
        return

    # ── Single image ───────────────────────────────────────────────────────
    if args.sketch:
        try:
            sketch = load_sketch(args.sketch)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"[Error] {e}")
            sys.exit(1)

        print(f"  Processing: {args.sketch}")
        try:
            result, ctrl = generate(sketch, pipeline, extractor, args)
        except RuntimeError as e:
            print(f"[Error] {e}")
            sys.exit(1)

        try:
            save_result(result, args.output)
            print(f"  ✓ Saved → {args.output}")
        except (PermissionError, OSError) as e:
            print(f"[Error] {e}")
            sys.exit(1)

        if args.save_comparison:
            cmp_path = args.output.replace(".png", "_comparison.png")
            try:
                save_comparison(
                    sketch.resize((args.img_size,) * 2),
                    ctrl, result, cmp_path
                )
                print(f"  ✓ Comparison → {cmp_path}")
            except Exception as e:
                print(f"  [Warning] Could not save comparison: {e}")
        return

    # ── Batch mode ─────────────────────────────────────────────────────────
    if args.sketch_dir:
        if not os.path.isdir(args.sketch_dir):
            print(f"[Error] --sketch_dir not found: '{args.sketch_dir}'")
            sys.exit(1)

        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except PermissionError:
            print(f"[Error] Cannot create output dir: '{args.output_dir}'")
            sys.exit(1)

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted(
            p for p in Path(args.sketch_dir).iterdir()
            if p.suffix.lower() in valid_exts
        )

        if not paths:
            print(f"[Error] No images found in '{args.sketch_dir}'")
            sys.exit(1)

        print(f"  Found {len(paths)} sketches in: {args.sketch_dir}")
        success_count = 0
        fail_count    = 0

        for sp in tqdm(paths, desc="Generating"):
            try:
                sketch = load_sketch(str(sp))
                result, ctrl = generate(sketch, pipeline, extractor, args)
                out = Path(args.output_dir) / f"{sp.stem}_gen.png"
                save_result(result, str(out))

                if args.save_comparison:
                    cmp = Path(args.output_dir) / f"{sp.stem}_cmp.png"
                    try:
                        save_comparison(
                            sketch.resize((args.img_size,) * 2),
                            ctrl, result, str(cmp)
                        )
                    except Exception as e:
                        print(f"  [Warning] Comparison save failed for {sp.name}: {e}")

                success_count += 1

            except FileNotFoundError as e:
                print(f"  [Skip] {sp.name}: {e}")
                fail_count += 1
            except RuntimeError as e:
                print(f"  [Skip] {sp.name}: {e}")
                fail_count += 1
            except KeyboardInterrupt:
                print(f"\n  Interrupted. Processed {success_count}/{len(paths)} images.")
                sys.exit(0)
            except Exception as e:
                print(f"  [Skip] {sp.name}: Unexpected error — {e}")
                fail_count += 1

        print(f"\n  ✓ Done: {success_count} succeeded, {fail_count} failed.")
        print(f"  Output → {args.output_dir}")
        return

    print("[Error] Specify --sketch, --sketch_dir, or --demo")
    print("  Examples:")
    print("    python inference.py --sketch cat.png --output result.png")
    print("    python inference.py --sketch_dir ./sketches --output_dir ./results")
    print("    python inference.py --demo")
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        traceback.print_exc()
        sys.exit(1)