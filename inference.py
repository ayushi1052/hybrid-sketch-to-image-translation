"""
inference.py
=============
SGLDv2 Inference Script

Modes:
  --sketch      : single sketch image → generated photo
  --sketch_dir  : folder of sketches  → folder of results
  --demo        : Gradio web UI

Usage
-----
  python inference.py --sketch ./cat.png --output ./result.png

  python inference.py --sketch_dir ./dataset/sketch/aeroplane \
                      --output_dir ./results/aeroplane

  python inference.py --demo --adapter ./checkpoints/final
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from models         import SGLDv2InferencePipeline, StructureExtractor
from utils          import set_seed, save_comparison


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SGLDv2 Inference")

    # Model paths
    p.add_argument("--adapter",     type=str, default="./checkpoints/final",
                   help="Path to trained SketchAdapter checkpoint")
    p.add_argument("--base_model",  type=str,
                   default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--num_tokens",  type=int, default=16)
    p.add_argument("--attn_scale",  type=float, default=1.0)
    p.add_argument("--img_size",    type=int, default=256)

    # Input / output
    p.add_argument("--sketch",      type=str, default=None)
    p.add_argument("--output",      type=str, default="./output.png")
    p.add_argument("--sketch_dir",  type=str, default=None)
    p.add_argument("--output_dir",  type=str, default="./results")
    p.add_argument("--demo",        action="store_true")

    # Generation parameters
    p.add_argument("--prompt",      type=str,
                   default="a realistic photo, highly detailed, sharp focus")
    p.add_argument("--neg_prompt",  type=str,
                   default="low quality, blurry, sketch, line drawing, cartoon")
    p.add_argument("--steps",       type=int,   default=30)
    p.add_argument("--guidance",    type=float, default=7.5)
    p.add_argument("--seed",        type=int,   default=42)

    # Options
    p.add_argument("--category",    type=str, default=None,
                   help="Category name for auto-caption (e.g. 'aeroplane')")
    p.add_argument("--no_structure",action="store_true")
    p.add_argument("--save_comparison", action="store_true")
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Core generation
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    sketch_pil:  Image.Image,
    pipeline:    SGLDv2InferencePipeline,
    extractor:   StructureExtractor,
    args:        argparse.Namespace,
    prompt:      str = None,
) -> tuple:
    """
    Run SGLDv2 on a single sketch.
    Returns: (generated_image, structure_map_pil)
    """
    prompt = prompt or args.prompt
    size   = args.img_size

    sketch_np = np.array(sketch_pil.convert("RGB").resize((size, size)))

    # Extract structure maps → 5-channel tensor
    if extractor is not None:
        structure = extractor.extract(sketch_np, (size, size))  # (5, H, W)
        ctrl_pil  = extractor.build_control_pil(sketch_np, size, "combined")
    else:
        from torchvision import transforms
        t = transforms.ToTensor()(
            Image.fromarray(sketch_np)
        )                                             # (3, H, W)
        structure = torch.cat([t, torch.zeros(2, size, size)], dim=0)
        ctrl_pil  = sketch_pil.resize((size, size))

    result = pipeline.generate(
        structure_maps  = structure,
        prompt          = prompt,
        negative_prompt = args.neg_prompt,
        guidance_scale  = args.guidance,
        num_steps       = args.steps,
        seed            = args.seed,
        width           = size,
        height          = size,
    )
    return result, ctrl_pil


# ─────────────────────────────────────────────────────────────────────────────
# Gradio Demo
# ─────────────────────────────────────────────────────────────────────────────

def launch_demo(
    pipeline:  SGLDv2InferencePipeline,
    extractor: StructureExtractor,
    args:      argparse.Namespace,
):
    """Interactive Gradio demo."""
    import gradio as gr
    from dataset_loader import build_caption

    def on_generate(sketch_np, prompt, neg_prompt,
                    guidance, steps, seed, category):
        if sketch_np is None:
            return None, None, "⚠️ Please upload a sketch."
        try:
            sketch_pil = Image.fromarray(sketch_np).convert("RGB")
            auto_prompt = build_caption(category) if category else prompt

            result, ctrl = generate(
                sketch_pil, pipeline, extractor,
                args, prompt=auto_prompt
            )
            return ctrl, result, "✓ Done"
        except Exception as e:
            return None, None, f"❌ {e}"

    def on_preview(sketch_np, mode):
        if sketch_np is None or extractor is None:
            return None
        sk = np.array(
            Image.fromarray(sketch_np).convert("RGB").resize(
                (args.img_size, args.img_size)
            )
        )
        return extractor.build_control_pil(sk, args.img_size, mode)

    with gr.Blocks(title="SGLDv2: Sketch → Photo",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## 🎨 SGLDv2 — Structure-Guided Latent Diffusion v2\n"
            "Sketch → Realistic Photo  |  SD fully frozen  |  "
            "Beats D-Sketch on structure fidelity"
        )
        with gr.Row():
            with gr.Column():
                sketch_in = gr.Image(label="✏️ Input Sketch",
                                     type="numpy", height=350)
                category  = gr.Textbox(
                    label="Category (optional — auto-generates prompt)",
                    placeholder="e.g.  aeroplane, apple, ball"
                )
                with gr.Row():
                    mode_dd    = gr.Dropdown(
                        ["edge","depth","combined"], value="combined",
                        label="Preview Mode"
                    )
                    preview_btn= gr.Button("👁 Preview Structure")

            with gr.Column():
                ctrl_out  = gr.Image(label="📐 Structure Map",
                                     type="pil", height=170)
                result_out= gr.Image(label="🖼 Generated Photo",
                                     type="pil", height=350)

        with gr.Row():
            prompt    = gr.Textbox(
                label="Prompt",
                value="a realistic photo, highly detailed, sharp focus, 8k",
                lines=2, scale=3
            )
            neg_p     = gr.Textbox(
                label="Negative Prompt",
                value="low quality, blurry, sketch, cartoon, line drawing",
                lines=2, scale=3
            )
        with gr.Row():
            guidance  = gr.Slider(1.0, 20.0, value=7.5, step=0.5,
                                  label="Guidance Scale")
            steps     = gr.Slider(10, 50, value=30, step=5, label="Steps")
            seed      = gr.Number(value=42, label="Seed", precision=0)

        status    = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            run_btn   = gr.Button("🚀 Generate", variant="primary", scale=3)
            clear_btn = gr.Button("🗑 Clear",    variant="stop",    scale=1)

        run_btn.click(
            fn      = on_generate,
            inputs  = [sketch_in, prompt, neg_p, guidance, steps, seed, category],
            outputs = [ctrl_out, result_out, status],
        )
        preview_btn.click(
            fn      = on_preview,
            inputs  = [sketch_in, mode_dd],
            outputs = [ctrl_out],
        )
        clear_btn.click(
            fn      = lambda: (None, None, None, ""),
            outputs = [sketch_in, ctrl_out, result_out, status],
        )

    demo.launch(share=True, server_port=7860)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # Load pipeline
    pipeline  = SGLDv2InferencePipeline(
        adapter_path = args.adapter,
        base_model   = args.base_model,
        device       = args.device,
        num_tokens   = args.num_tokens,
        img_size     = args.img_size,
        attn_scale   = args.attn_scale,
    )
    extractor = None if args.no_structure else StructureExtractor(args.device)

    # Auto-prompt from category
    if args.category:
        from dataset_loader import build_caption
        args.prompt = build_caption(args.category)
        print(f"Auto-prompt: {args.prompt}")

    # ── Demo ──────────────────────────────────────────────────────────────
    if args.demo:
        launch_demo(pipeline, extractor, args)
        return

    # ── Single image ───────────────────────────────────────────────────────
    if args.sketch:
        sketch = Image.open(args.sketch).convert("RGB")
        print(f"Processing: {args.sketch}")
        result, ctrl = generate(sketch, pipeline, extractor, args)
        result.save(args.output)
        print(f"✓ Saved → {args.output}")
        if args.save_comparison:
            cmp = args.output.replace(".png", "_comparison.png")
            save_comparison(
                sketch.resize((args.img_size, args.img_size)),
                ctrl, result, cmp
            )
            print(f"✓ Comparison → {cmp}")
        return

    # ── Batch ──────────────────────────────────────────────────────────────
    if args.sketch_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        exts = {".jpg",".jpeg",".png",".webp"}
        paths = sorted(
            p for p in Path(args.sketch_dir).iterdir()
            if p.suffix.lower() in exts
        )
        print(f"Found {len(paths)} sketches in {args.sketch_dir}")

        for sp in tqdm(paths, desc="Generating"):
            sketch = Image.open(sp).convert("RGB")
            result, ctrl = generate(sketch, pipeline, extractor, args)

            out = Path(args.output_dir) / f"{sp.stem}_generated.png"
            result.save(out)
            if args.save_comparison:
                cmp = Path(args.output_dir) / f"{sp.stem}_comparison.png"
                save_comparison(
                    sketch.resize((args.img_size, args.img_size)),
                    ctrl, result, str(cmp)
                )
        print(f"\n✓ Results saved to: {args.output_dir}")
        return

    print("Specify --sketch, --sketch_dir, or --demo")


if __name__ == "__main__":
    main()
