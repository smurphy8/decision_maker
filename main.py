from __future__ import annotations

import argparse
import sys
from typing import Optional

from decision_maker import DecisionMaker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a one-shot prompt against a Hugging Face causal LM."
    )
    parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="Model identifier passed to transformers.AutoModelForCausalLM.from_pretrained",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt text. If omitted, read from STDIN or fall back to a default.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate beyond the prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature; set to 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability mass.",
    )
    parser.add_argument(
        "--return-full-text",
        action="store_true",
        help="Return both prompt and completion text instead of just the completion.",
    )
    parser.add_argument(
        "--export-torchscript",
        metavar="PATH",
        help="If set, export a scripted DecisionMaker module to the given .pt path.",
    )
    parser.add_argument(
        "--trace-sequence-length",
        type=int,
        default=8,
        help="Sequence length used when tracing the module for torchscript export.",
    )
    parser.add_argument(
        "--trace-batch-size",
        type=int,
        default=1,
        help="Batch size used when tracing the module for torchscript export.",
    )
    parser.add_argument(
        "--trace-without-mask",
        action="store_true",
        help="Skip providing an attention mask when tracing for torchscript export.",
    )
    parser.add_argument(
        "--device",
        help="Optional torch device (cpu, cuda, cuda:0, etc.). Defaults to auto-select.",
    )
    parser.add_argument(
        "--dtype",
        dest="torch_dtype",
        help="Optional torch dtype string (float32, float16, bfloat16).",
    )
    return parser


def resolve_prompt(explicit_prompt: Optional[str]) -> str:
    if explicit_prompt:
        return explicit_prompt
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data
    return "You are a decision maker AI. Respond succinctly."


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    prompt = resolve_prompt(args.prompt)
    maker = DecisionMaker(
        args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    if args.export_torchscript:
        scripted = maker.as_torchscript(
            sequence_length=args.trace_sequence_length,
            batch_size=args.trace_batch_size,
            use_attention_mask=not args.trace_without_mask,
        )
        scripted.save(args.export_torchscript)

    result = maker.generate_from_prompt(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        return_full_text=args.return_full_text,
    )

    print(result.text)


if __name__ == "__main__":
    main()
