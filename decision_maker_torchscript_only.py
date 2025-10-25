from __future__ import annotations

"""
TorchScript-only runner that wires a scripted tokenizer with a traced model.

Goals:
- Avoid heavy dependencies. Only requires PyTorch and the two .pt files.
- Encode UTF-8 text to bytes → tokenizer → tensors → model forward.
- Be tolerant to shape/signature differences (with/without attention_mask).
- Optional greedy stepping using the model's final-position logits.

Assumptions:
- The tokenizer .pt is a TorchScript module whose forward accepts a 1D Long
  tensor of byte values (0..255) and returns either:
    (input_ids: 1D Long, attention_mask: 1D Long)
  or just input_ids (1D Long). This matches `tokenizer_ts.py` in this repo.
- The model .pt is a TorchScript module traced from DecisionMaker that takes
  (input_ids[, attention_mask]) and returns logits for the final position with
  shape [batch, vocab_size].

If the artifacts were exported with different conventions, the script tries
to adapt by probing call signatures and falling back where reasonable.
"""

import argparse
import sys
from typing import Optional, Tuple, List

import torch


def _auto_device(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_ts(path: str, device: str):
    mod = torch.jit.load(path, map_location=device)
    try:
        mod.eval()
    except Exception:
        pass
    try:
        mod.to(device)
    except Exception:
        # Many scripted modules don't need an explicit .to()
        pass
    return mod


def _module_device(mod) -> torch.device:
    try:
        p = next(mod.parameters())
        return p.device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encode_with_tokenizer(
    tok, text: str, max_length: int, add_eos: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Attempt several tokenizer signatures to obtain 1D ids/mask.

    Tries in order:
      1) tok(bytes_1d, max_length:int, add_eos:bool)
      2) tok(bytes_1d, max_length:int)
      3) tok(input_ids_1d, attention_mask_1d) with placeholder tensors
      4) Fallback to placeholder tensors without calling tokenizer
    """
    b = text.encode("utf-8")
    byte_tensor = torch.tensor(list(b), dtype=torch.long)

    with torch.no_grad():
        # Try byte-based calls first
        try:
            out = tok(byte_tensor, int(max_length), bool(add_eos))
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                ids = out[0]
                mask = out[1]
            elif torch.is_tensor(out):
                ids = out
                mask = (ids != 0).to(dtype=torch.long)
            else:
                raise RuntimeError
            if ids.dim() == 2 and ids.size(0) == 1:
                ids = ids.squeeze(0)
            if mask is not None and mask.dim() == 2 and mask.size(0) == 1:
                mask = mask.squeeze(0)
            if ids.dim() != 1:
                raise RuntimeError
            if mask is not None and mask.dim() != 1:
                raise RuntimeError
            return ids, mask
        except Exception:
            try:
                out = tok(byte_tensor, int(max_length))
                if isinstance(out, (list, tuple)) and len(out) >= 2:
                    ids = out[0]
                    mask = out[1]
                elif torch.is_tensor(out):
                    ids = out
                    mask = (ids != 0).to(dtype=torch.long)
                else:
                    raise RuntimeError
                if ids.dim() == 2 and ids.size(0) == 1:
                    ids = ids.squeeze(0)
                if mask is not None and mask.dim() == 2 and mask.size(0) == 1:
                    mask = mask.squeeze(0)
                if ids.dim() != 1:
                    raise RuntimeError
                if mask is not None and mask.dim() != 1:
                    raise RuntimeError
                return ids, mask
            except Exception:
                pass

    # Try tensor-tensor signature (e.g., InfernoDecisionTokenizer)
    ids_ph = torch.zeros(max_length, dtype=torch.long)
    if max_length > 0:
        ids_ph[-1] = 1
    mask_ph = torch.ones(max_length, dtype=torch.long)
    with torch.no_grad():
        try:
            out = tok(ids_ph, mask_ph)
            if isinstance(out, (list, tuple)) and len(out) >= 2 and torch.is_tensor(out[0]):
                ids = out[0]
                mask = out[1] if len(out) > 1 and torch.is_tensor(out[1]) else mask_ph
            elif torch.is_tensor(out):
                ids = out
                mask = mask_ph
            else:
                raise RuntimeError
            if ids.dim() == 2 and ids.size(0) == 1:
                ids = ids.squeeze(0)
            if mask is not None and mask.dim() == 2 and mask.size(0) == 1:
                mask = mask.squeeze(0)
            if ids.dim() != 1:
                raise RuntimeError
            if mask is not None and mask.dim() != 1:
                raise RuntimeError
            return ids, mask
        except Exception:
            pass

    # Final fallback: use placeholders directly
    return ids_ph, mask_ph


def _call_model(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        try:
            if attention_mask is None:
                return model(input_ids)
            return model(input_ids, attention_mask)
        except (TypeError, RuntimeError) as e:
            msg = str(e)
            # Fallback to fewer args if the artifact was traced without mask
            if "but got" in msg or "missing value for argument 'attention_mask'" in msg:
                return model(input_ids)
            raise


def _greedy_steps(
    model,
    input_ids: torch.Tensor,  # [1, L]
    attention_mask: Optional[torch.Tensor],  # [1, L] or None
    steps: int,
    eos_id: int,
) -> List[int]:
    """Do simple greedy generation by sliding the window and appending tokens.

    Returns a list of the newly generated token ids (length <= steps).
    """
    generated: List[int] = []
    if steps <= 0:
        return generated

    for _ in range(steps):
        logits = _call_model(model, input_ids, attention_mask)
        next_id_t = torch.argmax(logits, dim=-1)
        if next_id_t.dim() > 0:
            next_id = int(next_id_t.view(-1)[0].item())
        else:
            next_id = int(next_id_t.item())
        generated.append(next_id)

        # Stop early on EOS if provided
        if eos_id >= 0 and next_id == eos_id:
            break

        # Slide window left and append new id at end
        input_ids = torch.roll(input_ids, shifts=-1, dims=1)
        input_ids[:, -1] = next_id
        if attention_mask is not None:
            attention_mask = torch.roll(attention_mask, shifts=-1, dims=1)
            attention_mask[:, -1] = 1

    return generated


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a TorchScript tokenizer + model pair with minimal deps",
    )
    p.add_argument("--tokenizer", default="tokenizer.pt", help="Path to TorchScript tokenizer .pt")
    p.add_argument("--model", default="decision_maker_inferno.pt", help="Path to TorchScript model .pt")
    p.add_argument("--input", dest="text", help="Input text. If omitted, read from STDIN or use a default.")
    p.add_argument("--max-length", type=int, default=8, help="Max sequence length for tokenizer (must match trace)")
    p.add_argument("--steps", type=int, default=0, help="Greedy generation steps beyond the prompt window")
    p.add_argument("--device", help="torch device (cpu, cuda, cuda:0). Defaults to auto-detect")
    p.add_argument("--no-mask", action="store_true", help="Force calling model without attention_mask")
    p.add_argument("--eos-id", type=int, default=-1, help="EOS id for early stopping (negative = disabled)")
    return p


def _resolve_text(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data:
            return data
    return "Hello from TorchScript-only runner."


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    device = _auto_device(args.device)

    # Load artifacts
    tok = _load_ts(args.tokenizer, device="cpu")  # tokenizer runs fine on CPU
    model = _load_ts(args.model, device=device)
    model_dev = _module_device(model)

    text = _resolve_text(args.text)
    input_ids_1d, attn_1d = _encode_with_tokenizer(tok, text, args.max_length, add_eos=True)

    # Batchify and move to model device
    input_ids = input_ids_1d.unsqueeze(0).to(model_dev)
    attention_mask = None if attn_1d is None else attn_1d.unsqueeze(0).to(model_dev)

    logits = _call_model(model, input_ids, None if args.no_mask else attention_mask)
    top_id = int(torch.argmax(logits, dim=-1).view(-1)[0].item())

    # Best-effort EOS discovery from tokenizer attribute if not provided
    eos_id = int(args.eos_id)
    if eos_id < 0:
        try:
            eos_attr = getattr(tok, "eos_id")
            if isinstance(eos_attr, int):
                eos_id = int(eos_attr)
        except Exception:
            pass

    gen_ids: List[int] = []
    if args.steps > 0:
        gen_ids = _greedy_steps(
            model,
            input_ids.clone(),
            (None if args.no_mask else (attention_mask.clone() if attention_mask is not None else None)),
            args.steps,
            eos_id,
        )

    # Print a compact, dependency-free summary
    print("tokenizer:", args.tokenizer)
    print("model:", args.model)
    print("device:", model_dev)
    print("input_ids shape:", tuple(input_ids.shape))
    print("logits shape:", tuple(logits.shape))
    print("next_token_id:", top_id)
    if gen_ids:
        print("generated_ids:", gen_ids)


if __name__ == "__main__":
    main(sys.argv[1:])
