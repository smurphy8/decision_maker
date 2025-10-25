"""
InfernoML-oriented export entrypoint for the DecisionMaker model.

This uses DecisionMaker.as_torchscript(), which safely traces only the
forward path and applies the masking patch needed for HF models, avoiding
TorchScript trying to compile transformers configuration code.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

import torch

from decision_maker import DecisionMaker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a TorchScript forward-only module for InfernoML and optionally run a quick forward test.",
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Model identifier for transformers.AutoModelForCausalLM.from_pretrained",
    )
    parser.add_argument(
        "--device",
        help="Optional torch device (cpu, cuda, cuda:0). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--dtype",
        dest="torch_dtype",
        help="Optional torch dtype string (float32, float16, bfloat16)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Default max_new_tokens to configure the DecisionMaker (not used during tracing).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Default temperature to configure the DecisionMaker (not used during tracing).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Default nucleus sampling p for DecisionMaker (not used during tracing).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8,
        help="Sequence length to use when tracing. Keep consistent for inference with the traced artifact.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to use when tracing. Keep consistent for inference with the traced artifact.",
    )
    parser.add_argument(
        "--without-mask",
        action="store_true",
        help="Skip providing an attention_mask when tracing (by default a mask is used).",
    )
    parser.add_argument(
        "--output",
        default="decision_maker_inferno.pt",
        help="Path to save the traced TorchScript module.",
    )
    parser.add_argument(
        "--export-tokenizer",
        help=(
            "Optional path to save a TorchScript ByteLevel BPE tokenizer sidecar. "
            "Uses --tokenizer assets (default: ./tokenizer_config)."
        ),
    )
    parser.add_argument(
        "--test-forward",
        action="store_true",
        help="After export, load the artifact and run a quick forward pass.",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer_config",
        help="Local tokenizer path to use for --test-forward (defaults to ./tokenizer_config).",
    )
    return parser


def run_test_forward(
    artifact_path: str,
    tokenizer_path: str,
    *,
    sequence_length: int,
    batch_size: int,
    device: Optional[str],
) -> None:
    """Load the traced module and run a simple forward to verify shapes.

    This uses a local tokenizer path to avoid network access. If the tokenizer
    cannot be loaded, decoding of the top token will be skipped but the forward
    run and logits shape will still be printed.
    """
    m = torch.jit.load(artifact_path, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    m.eval()
    if device:
        try:
            m.to(device)
        except Exception:
            pass
    # Determine the device the module resides on for input placement.
    try:
        first_param = next(m.parameters())
        m_device = first_param.device
    except StopIteration:
        m_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tok = None
    top_decoded: Optional[str] = None
    try:
        from transformers import AutoTokenizer  # imported lazily

        tok = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"[inferno] Warning: could not load tokenizer from '{tokenizer_path}': {e}")

    test_text = "Hello from Inferno"
    if tok is not None:
        enc = tok(
            test_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        use_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
    else:
        input_ids = torch.zeros((batch_size, sequence_length), dtype=torch.long)
        input_ids[:, -1] = 1  # minimal non-zero to avoid full padding
        use_mask = torch.ones_like(input_ids)

    # Move inputs to the same device as the module to avoid device mismatch.
    input_ids = input_ids.to(m_device)
    if isinstance(use_mask, torch.Tensor):
        use_mask = use_mask.to(m_device)

    with torch.no_grad():
        try:
            logits = m(input_ids, use_mask)
        except (TypeError, RuntimeError) as e:
            # Fallback for artifacts traced without an attention_mask input.
            msg = str(e)
            if "missing value for argument 'attention_mask'" in msg or "but got 2" in msg:
                logits = m(input_ids)
            else:
                raise
    print("[inferno] logits shape:", tuple(logits.shape))
    try:
        top_id = int(logits.argmax(dim=-1).item())
        if tok is not None:
            top_decoded = tok.decode([top_id])
    except Exception:
        pass
    if top_decoded is not None:
        print(f"[inferno] next token id: {top_id} -> {top_decoded}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    maker = DecisionMaker(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    # Use the built-in helper that traces a thin forward-only wrapper over the
    # underlying HF model. This avoids TorchScript attempting to compile
    # transformers configs (e.g., Qwen2Config.__init__(**kwargs)).
    traced = maker.as_torchscript(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        use_attention_mask=not args.without_mask,
    )
    traced.save(args.output)
    print(f"Saved traced model to {args.output}")

    # Optionally export a TorchScript-friendly tokenizer built from local assets
    if args.export_tokenizer:
        try:
            from tokenizer_ts import build_scripted_tokenizer  # local module

            tok = build_scripted_tokenizer(args.tokenizer)
            tok.save(args.export_tokenizer)
            print(f"Saved scripted tokenizer to {args.export_tokenizer}")
        except Exception as e:
            print(f"[inferno] Warning: failed to export tokenizer: {e}")

    if args.test_forward:
        run_test_forward(
            args.output,
            args.tokenizer,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            device=args.device,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
