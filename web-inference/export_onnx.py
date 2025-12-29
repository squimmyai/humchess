"""
Export HumChess model to ONNX format for browser inference.

Usage:
    uv run python web-inference/export_onnx.py --checkpoint checkpoints/epoch_1.pt --output web-inference/static/models/humchess.onnx
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.onnx

from humchess.model.transformer import ChessTransformer


def export_onnx(checkpoint_path: Path, output_path: Path, quantize: bool = False):
    """Export model to ONNX format."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Infer model config from state dict
    state_dict = checkpoint["model_state_dict"]
    d_model = state_dict["token_emb.weight"].shape[1]
    n_layers = sum(1 for k in state_dict.keys() if k.startswith("blocks.") and k.endswith(".rmsnorm1.weight"))
    d_ff = state_dict["blocks.0.ffn.w1.weight"].shape[0]

    # Infer n_heads from QK-norm weight shape (it's per-head, so shape is head_dim)
    head_dim = state_dict["blocks.0.attn.q_norm.weight"].shape[0]
    n_heads = d_model // head_dim

    print(f"Inferred config: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, d_ff={d_ff}")

    # Create model and load weights
    model = ChessTransformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model parameters: {model.count_parameters():,}")

    # Create dummy input (int64 for embedding lookup)
    dummy_input = torch.zeros(1, 68, dtype=torch.long)

    # Export to ONNX
    print(f"Exporting to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the legacy exporter to get a single self-contained file
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["tokens"],
        output_names=["move_logits", "promo_logits"],
        dynamic_axes={
            "tokens": {0: "batch"},
            "move_logits": {0: "batch"},
            "promo_logits": {0: "batch"},
        },
        opset_version=14,  # Use older opset for better browser compatibility
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter for single-file output
    )

    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported ONNX model: {size_mb:.2f} MB")

    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = output_path.with_suffix(".quantized.onnx")
            print(f"Quantizing to: {quantized_path}")

            quantize_dynamic(
                str(output_path),
                str(quantized_path),
                weight_type=QuantType.QUInt8,
            )

            quant_size_mb = quantized_path.stat().st_size / (1024 * 1024)
            print(f"Quantized model: {quant_size_mb:.2f} MB")
        except ImportError:
            print("Warning: onnxruntime not installed, skipping quantization")

    # Verify the export
    print("\nVerifying ONNX model...")
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # Test inference
    print("\nTesting inference with ONNX Runtime...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path))
    test_input = dummy_input.numpy()
    outputs = session.run(None, {"tokens": test_input})

    print(f"Move logits shape: {outputs[0].shape}")
    print(f"Promo logits shape: {outputs[1].shape}")
    print("\nExport successful!")


def main():
    parser = argparse.ArgumentParser(description="Export HumChess model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX path")
    parser.add_argument("--quantize", action="store_true", help="Also create quantized version")
    args = parser.parse_args()

    export_onnx(Path(args.checkpoint), Path(args.output), args.quantize)


if __name__ == "__main__":
    main()
