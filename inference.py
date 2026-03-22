"""
inference.py
=============
SGLDv2 Inference — sketch → realistic photo

Usage
─────
  # Single image
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
    p = argparse.ArgumentParser(description="SGLDv2 sketch-to-image inference")

    p.add_argument("--adapter",         type=str, default="./checkpoints/final")
    p.add_argument("--base_model",      type=str,
                   default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--img_size",        type=int, default=256)
    p.add_argument("--device",          type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--sketch",          type=str, default=None)
    p.add_argument("--output",          type=str, default="./output.png")
    p.add_argument("--sketch_dir",      type=str, default=None)
    p.add_argument("--output_dir",      type=str, default="./results")
    p.add_argument("--save_comparison", action="store_true")
    p.add_argument("--demo",            action="store_true")

    p.add_argument("--prompt",          type=str,
                   default=SGLDv2InferencePipeline.FIXED_PROMPT)
    p.add_argument("--neg_prompt",      type=str,
                   default=SGLDv2InferencePipeline.FIXED_NEG_PROMPT)
    p.add_argument("--steps",           type=int,   default=30)
    p.add_argument("--guidance",        type=float, default=7.5)
    p.add_argument("--cond_scale",      type=float, default=1.0,
                   help="ControlNet conditioning scale (0.5–2.0)")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--no_structure",    action="store_true",
                   help="Skip StructureExtractor — use raw sketch as hint")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline(args: argparse.Namespace) -> SGLDv2InferencePipeline:
    if not os.path.isdir(args.adapter):
        print(f"[Error] Checkpoint not found: '{args.adapter}'")
        print("        Train first:  python train.py --data_dir ./dataset")
        sys.exit(1)
    try:
        return SGLDv2InferencePipeline(
            adapter_path = args.adapter,
            base_model   = args.base_model,
            device       = args.device,
            img_size     = args.img_size,
        )
    except OSError as e:
        print(f"[Error] Could not load '{args.base_model}': {e}")
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        print("[Error] GPU OOM. Try: --device cpu")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Pipeline load failed: {e}")
        traceback.print_exc()
        sys.exit(1)


def load_extractor(args: argparse.Namespace) -> StructureExtractor | None:
    if args.no_structure:
        return None
    try:
        # use_cache=False at inference — no disk writes for one-off runs
        return StructureExtractor(device=args.device, use_cache=False)
    except Exception as e:
        print(f"  [Warning] StructureExtractor failed ({e}). Raw sketch mode.")
        return None


def load_sketch(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sketch not found: '{path}'")
    valid = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if Path(path).suffix.lower() not in valid:
        raise ValueError(f"Unsupported format: '{Path(path).suffix}'")
    try:
        img = Image.open(path).convert("RGB")
        if img.size[0] < 16 or img.size[1] < 16:
            raise ValueError(f"Image too small: {img.size}")
        return img
    except (OSError, IOError) as e:
        raise IOError(f"Cannot open '{path}': {e}") from e


def build_structure(
    sketch_pil: Image.Image,
    extractor:  StructureExtractor | None,
    img_size:   int,
) -> tuple[torch.Tensor, Image.Image]:
    """Returns (structure_tensor (5,H,W), control_preview PIL)."""
    sk_np = np.array(sketch_pil.convert("RGB").resize((img_size, img_size)))
    if extractor is not None:
        try:
            struct   = extractor.extract(sk_np, (img_size, img_size))
            ctrl_pil = extractor.build_control_pil(sk_np, img_size, "combined")
            return struct, ctrl_pil
        except Exception as e:
            print(f"  [Warning] Structure extraction failed ({e}) — raw sketch.")
    # Fallback
    from torchvision import transforms
    t      = transforms.ToTensor()(Image.fromarray(sk_np))
    struct = torch.cat([t, torch.zeros(2, img_size, img_size)], dim=0)
    return struct, Image.fromarray(sk_np)


def generate(
    sketch_pil: Image.Image,
    pipeline:   SGLDv2InferencePipeline,
    extractor:  StructureExtractor | None,
    args:       argparse.Namespace,
    prompt:     str | None = None,
) -> tuple[Image.Image, Image.Image]:
    struct, ctrl_pil = build_structure(sketch_pil, extractor, args.img_size)
    try:
        result = pipeline.generate(
            structure_maps     = struct,
            prompt             = prompt or args.prompt,
            negative_prompt    = args.neg_prompt,
            guidance_scale     = args.guidance,
            num_steps          = args.steps,
            seed               = args.seed,
            conditioning_scale = args.cond_scale,
        )
        return result, ctrl_pil
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            "GPU OOM. Try: --steps 20 or --img_size 256 or --device cpu"
        )
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}") from e


def save_result(result: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    result.save(path)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio demo
# ─────────────────────────────────────────────────────────────────────────────

def launch_demo(
    pipeline:  SGLDv2InferencePipeline,
    extractor: StructureExtractor | None,
    args:      argparse.Namespace,
) -> None:
    try:
        import gradio as gr
    except ImportError:
        print("[Error] Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    def on_generate(sketch_np, prompt, neg, guidance, steps, seed, cond_scale):
        if sketch_np is None:
            return None, None, "⚠️ Upload a sketch first."
        try:
            pil = Image.fromarray(sketch_np).convert("RGB")
            if pil.size[0] < 16 or pil.size[1] < 16:
                return None, None, "⚠️ Image too small (min 16×16)."
            _a           = argparse.Namespace(**vars(args))
            _a.prompt    = prompt
            _a.neg_prompt = neg
            _a.guidance  = float(guidance)
            _a.steps     = int(steps)
            _a.seed      = int(seed)
            _a.cond_scale = float(cond_scale)
            ctrl, result = generate(pil, pipeline, extractor, _a)
            return ctrl, result, "✓ Done"
        except RuntimeError as e:
            return None, None, f"❌ {e}"
        except Exception as e:
            traceback.print_exc()
            return None, None, f"❌ {e}"

    def on_preview(sketch_np, mode):
        if sketch_np is None or extractor is None:
            return Image.fromarray(sketch_np) if sketch_np is not None else None
        try:
            sk = np.array(
                Image.fromarray(sketch_np).convert("RGB")
                .resize((args.img_size, args.img_size))
            )
            return extractor.build_control_pil(sk, args.img_size, mode)
        except Exception as e:
            print(f"  [Demo] Preview failed: {e}")
            return None

    with gr.Blocks(title="SGLDv2", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## 🎨 SGLDv2 — Structure-Guided Latent Diffusion v2\n"
            "**Sketch → Realistic Photo**  "
            "_(Canny edges + MiDaS depth + SegFormer segmentation → ControlNet → SD)_"
        )
        with gr.Row():
            with gr.Column():
                sketch_in   = gr.Image(label="✏️ Input Sketch",   type="numpy", height=350)
                with gr.Row():
                    mode        = gr.Dropdown(
                        ["edge", "depth", "seg", "combined"],
                        value="combined", label="Structure Preview"
                    )
                    preview_btn = gr.Button("👁 Preview Structure")
            with gr.Column():
                ctrl_out    = gr.Image(label="📐 Structure Map", type="pil", height=170)
                result_out  = gr.Image(label="🖼 Generated",     type="pil", height=350)

        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                value=SGLDv2InferencePipeline.FIXED_PROMPT, lines=2, scale=3,
            )
            neg_p  = gr.Textbox(
                label="Negative Prompt",
                value=SGLDv2InferencePipeline.FIXED_NEG_PROMPT, lines=2, scale=3,
            )
        with gr.Row():
            guidance   = gr.Slider(1.0, 20.0, 7.5,  step=0.5,  label="Guidance Scale")
            steps      = gr.Slider(10,  50,   30,   step=5,    label="Steps")
            cond_scale = gr.Slider(0.1,  2.0,  1.0,  step=0.05, label="ControlNet Scale")
            seed       = gr.Number(42, label="Seed", precision=0)

        status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            gr.Button("🚀 Generate", variant="primary").click(
                on_generate,
                [sketch_in, prompt, neg_p, guidance, steps, seed, cond_scale],
                [ctrl_out, result_out, status],
            )
            gr.Button("🗑 Clear", variant="stop").click(
                lambda: (None, None, None, ""),
                outputs=[sketch_in, ctrl_out, result_out, status],
            )
        preview_btn.click(on_preview, [sketch_in, mode], ctrl_out)

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
    args = parse_args()
    set_seed(args.seed)

    pipeline  = load_pipeline(args)
    extractor = load_extractor(args)

    if args.demo:
        launch_demo(pipeline, extractor, args)
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
            result, ctrl = generate(sketch, pipeline, extractor, args)
        except RuntimeError as e:
            print(f"[Error] {e}")
            sys.exit(1)
        try:
            save_result(result, args.output)
            print(f"  ✓ Saved → {args.output}")
        except Exception as e:
            print(f"[Error] {e}")
            sys.exit(1)
        if args.save_comparison:
            cmp = args.output.replace(".png", "_comparison.png")
            try:
                save_comparison(sketch.resize((args.img_size,)*2), ctrl, result, cmp)
                print(f"  ✓ Comparison → {cmp}")
            except Exception as e:
                print(f"  [Warning] Comparison failed: {e}")
        return

    # ── Batch mode ─────────────────────────────────────────────────────────
    if args.sketch_dir:
        if not os.path.isdir(args.sketch_dir):
            print(f"[Error] --sketch_dir not found: '{args.sketch_dir}'")
            sys.exit(1)
        os.makedirs(args.output_dir, exist_ok=True)
        valid  = {".jpg", ".jpeg", ".png", ".webp"}
        paths  = sorted(p for p in Path(args.sketch_dir).iterdir()
                        if p.suffix.lower() in valid)
        if not paths:
            print(f"[Error] No images in '{args.sketch_dir}'")
            sys.exit(1)
        print(f"  Found {len(paths)} sketches.")
        ok = fail = 0
        for sp in tqdm(paths, desc="Generating"):
            try:
                sk     = load_sketch(str(sp))
                result, ctrl = generate(sk, pipeline, extractor, args)
                out    = Path(args.output_dir) / f"{sp.stem}_gen.png"
                save_result(result, str(out))
                if args.save_comparison:
                    cmp = Path(args.output_dir) / f"{sp.stem}_cmp.png"
                    try:
                        save_comparison(
                            sk.resize((args.img_size,)*2), ctrl, result, str(cmp)
                        )
                    except Exception:
                        pass
                ok += 1
            except KeyboardInterrupt:
                print(f"\n  Interrupted — {ok} done.")
                sys.exit(0)
            except Exception as e:
                print(f"  [Skip] {sp.name}: {e}")
                fail += 1
        print(f"\n  ✓ {ok} succeeded, {fail} failed.  Output → {args.output_dir}")
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
