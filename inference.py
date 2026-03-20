"""
inference.py
=============
Inference for the modified d-Sketch pipeline.

Usage
─────
  # Single sketch
  python inference.py --sketch cat.png --output result.png

  # Batch directory
  python inference.py --sketch_dir ./sketches --output_dir ./results

  # Interactive Gradio demo
  python inference.py --demo
"""

from __future__ import annotations

import os
import sys
import argparse
import traceback
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from models         import SketchToImagePipeline
from utils          import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Modified d-Sketch inference — sketch → realistic photo"
    )
    p.add_argument("--checkpoint",      type=str, default="./checkpoints/best",
                   help="Path to checkpoint directory")
    p.add_argument("--sd_model",        type=str,
                   default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--clip_model",      type=str,
                   default="openai/clip-vit-large-patch14")
    p.add_argument("--img_size",        type=int, default=256)
    p.add_argument("--device",          type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # Single image
    p.add_argument("--sketch",          type=str, default=None)
    p.add_argument("--output",          type=str, default="./output.png")

    # Batch
    p.add_argument("--sketch_dir",      type=str, default=None)
    p.add_argument("--output_dir",      type=str, default="./results")

    # Demo
    p.add_argument("--demo",            action="store_true")

    # Generation parameters
    p.add_argument("--steps",           type=int,   default=30)
    p.add_argument("--guidance",        type=float, default=3.0,
                   help="CFG scale. Lower than standard SD (3-5 recommended) "
                        "since image conditioning already provides strong signal")
    p.add_argument("--k_ratio",         type=float, default=0.8)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--save_colored",    action="store_true",
                   help="Also save the predicted colored edge map")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Load pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline(args: argparse.Namespace) -> SketchToImagePipeline:
    if not os.path.isdir(args.checkpoint):
        print(f"[Error] Checkpoint not found: '{args.checkpoint}'")
        print("        Train first: python train.py --data_dir ./dataset")
        sys.exit(1)
    try:
        pipe = SketchToImagePipeline(
            sd_model   = args.sd_model,
            clip_model = args.clip_model,
            img_size   = args.img_size,
            device     = args.device,
            k_ratio    = args.k_ratio,
        )
        pipe.load(args.checkpoint)
        pipe.eval()
        return pipe
    except OSError as e:
        print(f"[Error] Model load failed: {e}")
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        print("[Error] GPU OOM. Try: --device cpu")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Pipeline load failed: {e}")
        traceback.print_exc()
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Generate
# ─────────────────────────────────────────────────────────────────────────────

def load_sketch(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sketch not found: '{path}'")
    valid = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if Path(path).suffix.lower() not in valid:
        raise ValueError(f"Unsupported format: '{Path(path).suffix}'")
    img = Image.open(path).convert("RGB")
    if img.size[0] < 16 or img.size[1] < 16:
        raise ValueError(f"Image too small: {img.size}")
    return img


def run_generate(
    sketch_pil: Image.Image,
    pipeline:   SketchToImagePipeline,
    args:       argparse.Namespace,
) -> tuple[Image.Image, Image.Image]:
    """Returns (output_photo, colored_edge_map_preview)"""
    try:
        return pipeline.generate(
            sketch_pil     = sketch_pil,
            num_steps      = args.steps,
            seed           = args.seed,
            guidance_scale = args.guidance,
        )
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            "GPU OOM. Try: --steps 20 or --device cpu"
        )
    except Exception as e:
        # Print full traceback so we can see exactly which line fails
        traceback.print_exc()
        raise RuntimeError(f"Generation failed: {e}") from e


def save_result(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio Demo
# ─────────────────────────────────────────────────────────────────────────────

def launch_demo(
    pipeline: SketchToImagePipeline,
    args:     argparse.Namespace,
) -> None:
    try:
        import gradio as gr
    except ImportError:
        print("[Error] Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    def on_generate(sketch_np, steps, guidance, seed, k_ratio):
        if sketch_np is None:
            return None, None, "⚠️ Please upload a sketch."
        try:
            pil = Image.fromarray(sketch_np).convert("RGB")
            if pil.size[0] < 16 or pil.size[1] < 16:
                return None, None, "⚠️ Image too small."

            _args          = argparse.Namespace(**vars(args))
            _args.steps    = int(steps)
            _args.guidance = float(guidance)
            _args.seed     = int(seed)
            _args.k_ratio  = float(k_ratio)
            pipeline.k_ratio = _args.k_ratio

            result, colored = run_generate(pil, pipeline, _args)
            return colored, result, "✓ Done"
        except RuntimeError as e:
            return None, None, f"❌ {e}"
        except Exception as e:
            traceback.print_exc()
            return None, None, f"❌ {e}"

    with gr.Blocks(title="d-Sketch Modified", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## 🎨 Modified d-Sketch — Sketch → Realistic Photo\n"
            "**No text prompt needed.** Just draw a sketch.\n\n"
            "The model learns to predict what real-world edges and colors "
            "your sketch corresponds to, then uses SD to generate the photo."
        )
        with gr.Row():
            with gr.Column():
                sketch_in = gr.Image(label="✏️ Input Sketch",
                                     type="numpy", height=350)
                with gr.Row():
                    steps    = gr.Slider(10, 50, 30, step=5,    label="Steps")
                    guidance = gr.Slider(1.0, 10.0, 3.0, step=0.5,
                                        label="Guidance Scale")
                with gr.Row():
                    k_ratio  = gr.Slider(0.5, 0.95, 0.8, step=0.05,
                                         label="k ratio (structure↔creativity)")
                    seed     = gr.Number(42, label="Seed", precision=0)
            with gr.Column():
                colored_out = gr.Image(label="📐 Predicted Colored Edge Map",
                                       type="pil", height=170)
                result_out  = gr.Image(label="🖼 Generated Photo",
                                       type="pil", height=350)

        status = gr.Textbox(label="Status", interactive=False)
        gr.Markdown(
            "**k ratio**: controls balance between sketch faithfulness and "
            "photorealism.\n"
            "- Lower (0.5–0.7) = very close to sketch structure\n"
            "- Higher (0.8–0.95) = more photorealistic but looser structure"
        )
        with gr.Row():
            gr.Button("🚀 Generate", variant="primary").click(
                on_generate,
                [sketch_in, steps, guidance, seed, k_ratio],
                [colored_out, result_out, status],
            )
            gr.Button("🗑 Clear", variant="stop").click(
                lambda: (None, None, None, ""),
                outputs=[sketch_in, colored_out, result_out, status],
            )

    try:
        demo.launch(share=True, server_port=7860)
    except OSError:
        demo.launch(share=True)
    except Exception as e:
        print(f"[Error] Demo launch failed: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args     = parse_args()
    set_seed(args.seed)
    pipeline = load_pipeline(args)

    # ── Demo ───────────────────────────────────────────────────────────────
    if args.demo:
        launch_demo(pipeline, args)
        return

    # ── Single image ───────────────────────────────────────────────────────
    if args.sketch:
        try:
            sketch = load_sketch(args.sketch)
        except Exception as e:
            print(f"[Error] {e}")
            sys.exit(1)

        print(f"  Processing: {args.sketch}")
        try:
            result, colored = run_generate(sketch, pipeline, args)
        except RuntimeError as e:
            print(f"[Error] {e}")
            sys.exit(1)

        try:
            save_result(result, args.output)
            print(f"  ✓ Output      → {args.output}")
        except Exception as e:
            print(f"[Error] {e}")
            sys.exit(1)

        if args.save_colored:
            col_path = args.output.replace(".png", "_colored_map.png")
            try:
                save_result(colored, col_path)
                print(f"  ✓ Colored map → {col_path}")
            except Exception as e:
                print(f"  [Warning] Colored map save failed: {e}")
        return

    # ── Batch mode ─────────────────────────────────────────────────────────
    if args.sketch_dir:
        if not os.path.isdir(args.sketch_dir):
            print(f"[Error] --sketch_dir not found: '{args.sketch_dir}'")
            sys.exit(1)
        os.makedirs(args.output_dir, exist_ok=True)

        valid = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted(p for p in Path(args.sketch_dir).iterdir()
                       if p.suffix.lower() in valid)
        if not paths:
            print(f"[Error] No images in '{args.sketch_dir}'")
            sys.exit(1)

        print(f"  Found {len(paths)} sketches.")
        ok = fail = 0

        for sp in tqdm(paths, desc="Generating"):
            try:
                sketch = load_sketch(str(sp))
                result, colored = run_generate(sketch, pipeline, args)

                out = Path(args.output_dir) / f"{sp.stem}_result.png"
                save_result(result, str(out))

                if args.save_colored:
                    col = Path(args.output_dir) / f"{sp.stem}_colored.png"
                    save_result(colored, str(col))

                ok += 1
            except KeyboardInterrupt:
                print(f"\n  Interrupted — {ok} done.")
                sys.exit(0)
            except Exception as e:
                print(f"  [Skip] {sp.name}: {e}")
                fail += 1

        print(f"\n  ✓ {ok} succeeded, {fail} failed.")
        print(f"  Output → {args.output_dir}")
        return

    print("[Error] Specify --sketch, --sketch_dir, or --demo")
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        traceback.print_exc()
        sys.exit(1)