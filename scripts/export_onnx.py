#!/usr/bin/env python3
"""
Export and quantize a token classification model for transformers.js/ONNX runtime.

Usage examples:
  python3 scripts/export_onnx.py
  HF_TOKEN=... python3 scripts/export_onnx.py --push
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MODEL_ID = "AndrewAndrewsen/distilbert-secret-masker"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--out-dir", default="model", help="Output directory")
    parser.add_argument(
        "--quantized-dir",
        default="model_quantized",
        help="Output directory for quantized ONNX artifacts",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push ONNX export to Hugging Face hub (requires HF_TOKEN)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from transformers import AutoTokenizer
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover
        print(f"Missing dependencies. Install optimum[onnxruntime], transformers, huggingface_hub: {exc}", file=sys.stderr)
        return 2

    api = HfApi()
    try:
        api.model_info(args.model_id)
    except Exception:
        print(
            f"Model '{args.model_id}' was not found on Hugging Face. "
            "Create/upload the base model first, then rerun export.",
            file=sys.stderr,
        )
        return 1

    print(f"Exporting model '{args.model_id}' to ONNX...")
    ort_model = ORTModelForTokenClassification.from_pretrained(args.model_id, export=True)
    AutoTokenizer.from_pretrained(args.model_id)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ort_model.save_pretrained(out_dir)

    print("Quantizing ONNX model to int8 dynamic quantization...")
    quantizer = ORTQuantizer.from_pretrained(out_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
    quantized_dir = Path(args.quantized_dir)
    quantized_dir.mkdir(parents=True, exist_ok=True)
    quantizer.quantize(save_dir=quantized_dir, quantization_config=qconfig)

    print(f"Saved ONNX model to: {out_dir.resolve()}")
    print(f"Saved quantized ONNX model to: {quantized_dir.resolve()}")

    if args.push:
        if not args.token:
            print("--push requested but no token provided. Use --token or HF_TOKEN.", file=sys.stderr)
            return 2

        print(f"Pushing ONNX variant to Hugging Face Hub repo '{args.model_id}'...")
        ort_model.push_to_hub(args.model_id, token=args.token, subfolder="onnx")
        print("Push complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
