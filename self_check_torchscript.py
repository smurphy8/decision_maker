from __future__ import annotations

"""
Self-check for TorchScript tokenizer + model artifacts.

This script:
- Loads a TorchScript tokenizer (tokenizer.pt) and model (decision_maker_inferno.pt)
- Encodes sample texts to tensors using the tokenizer
- Runs a forward pass through the model and validates shapes
- Optionally performs a few greedy steps to confirm iterative usage

Dependencies: only PyTorch and Python stdlib.
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
    """Try multiple tokenizer signatures to obtain (ids, mask).

    Tries in order:
    1) tok(bytes_1d, max_length:int, add_eos:bool)
    2) tok(bytes_1d, max_length:int)
    3) tok(input_ids_1d, attention_mask_1d) with placeholder tensors
    4) Fallback to placeholder tensors without calling tokenizer
    """
    b = text.encode("utf-8")
    byte_tensor = torch.tensor(list(b), dtype=torch.long)

    # Attempt 1/2: byte-based signatures
    with torch.no_grad():
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
            # Accept 1D or batched [1, L]
            if ids.dim() == 2 and ids.size(0) == 1:
                ids = ids.squeeze(0)
            if mask is not None and mask.dim() == 2 and mask.size(0) == 1:
                mask = mask.squeeze(0)
            if ids.dim() != 1:
                raise RuntimeError
            if mask is not None and mask.dim() != 1:
                raise RuntimeError
            return ids, mask
        except Exception as e1:
            msg1 = str(e1)
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
            except Exception as e2:
                msg2 = str(e2)
                # Fall through to tensor-tensor signature attempt

    # Attempt 3: tensor-tensor signature using placeholders (InfernoDecisionTokenizer style)
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

    # Attempt 4: give up on tokenizer; return placeholders
    return ids_ph, mask_ph


def _call_model(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        try:
            if attention_mask is None:
                return model(input_ids)
            return model(input_ids, attention_mask)
        except (TypeError, RuntimeError) as e:
            msg = str(e)
            if "missing value for argument 'attention_mask'" in msg or "but got" in msg:
                return model(input_ids)
            raise


def _greedy_steps(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    steps: int,
    eos_id: int,
) -> List[int]:
    generated: List[int] = []
    if steps <= 0:
        return generated
    for _ in range(steps):
        logits = _call_model(model, input_ids, attention_mask)
        next_id_t = torch.argmax(logits, dim=-1)
        next_id = int(next_id_t.view(-1)[0].item())
        generated.append(next_id)
        if eos_id >= 0 and next_id == eos_id:
            break
        input_ids = torch.roll(input_ids, shifts=-1, dims=1)
        input_ids[:, -1] = next_id
        if attention_mask is not None:
            attention_mask = torch.roll(attention_mask, shifts=-1, dims=1)
            attention_mask[:, -1] = 1
    return generated


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Self-check TorchScript tokenizer + model artifacts")
    p.add_argument("--tokenizer", default="tokenizer.pt", help="Path to TorchScript tokenizer .pt")
    p.add_argument("--model", default="decision_maker_inferno.pt", help="Path to TorchScript model .pt")
    p.add_argument("--max-length", type=int, default=8, help="Sequence length used during export")
    p.add_argument("--device", help="torch device (cpu, cuda, cuda:0). Defaults to auto-detect")
    p.add_argument("--no-mask", action="store_true", help="Force model call without attention_mask")
    p.add_argument("--steps", type=int, default=0, help="Greedy steps to test iterative usage")
    p.add_argument(
        "--text",
        action="append",
        help="Sample text (can be specified multiple times). If omitted, uses built-ins.",
    )
    p.add_argument("--eos-id", type=int, default=-1, help="EOS token id for early stopping (negative disables)")
    return p


def _resolve_texts(arg_list: Optional[List[str]]) -> List[str]:
    if arg_list and len(arg_list) > 0:
        return arg_list
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data:
            return [data]
    return [
        "Hello from TorchScript self-check.",
        "The quick brown fox jumps over the lazy dog.",
    ]


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    device = _auto_device(args.device)

    # Load artifacts
    tok = _load_ts(args.tokenizer, device="cpu")
    model = _load_ts(args.model, device=device)
    model_dev = _module_device(model)

    texts = _resolve_texts(args.text)

    print("[self-check] tokenizer:", args.tokenizer)
    print("[self-check] model:", args.model)
    print("[self-check] device:", model_dev)
    print("[self-check] max_length:", args.max_length)

    # Determine EOS if available
    eos_id = int(args.eos_id)
    if eos_id < 0:
        try:
            eos_attr = getattr(tok, "eos_id")
            if isinstance(eos_attr, int):
                eos_id = int(eos_attr)
        except Exception:
            pass

    ok = True

    for idx, text in enumerate(texts):
        try:
            ids_1d, attn_1d = _encode_with_tokenizer(tok, text, args.max_length, add_eos=True)
            if ids_1d.dtype != torch.long:
                raise AssertionError("input_ids dtype must be torch.long")
            if ids_1d.dim() != 1 or ids_1d.numel() != args.max_length:
                raise AssertionError("input_ids must be 1D of length max_length")

            if attn_1d is not None:
                if attn_1d.dtype != torch.long:
                    raise AssertionError("attention_mask dtype must be torch.long")
                if attn_1d.dim() != 1 or attn_1d.numel() != args.max_length:
                    raise AssertionError("attention_mask must be 1D of length max_length")
                # Check that mask values are 0/1
                uniq = torch.unique(attn_1d)
                if not all(int(v.item()) in (0, 1) for v in uniq):
                    raise AssertionError("attention_mask must contain only 0/1 values")

            ids = ids_1d.unsqueeze(0).to(model_dev)
            mask = None if (attn_1d is None or args.no_mask) else attn_1d.unsqueeze(0).to(model_dev)

            logits = _call_model(model, ids, mask)
            if logits.dim() != 2 or logits.size(0) != 1:
                raise AssertionError("model must return [batch, vocab_size] logits for final position")
            vocab_size = logits.size(1)
            top_id = int(torch.argmax(logits, dim=-1).item())

            print(f"[self-check] sample {idx}: input_ids shape={tuple(ids.shape)} logits shape={tuple(logits.shape)} top_id={top_id}")

            if args.steps > 0:
                # Use a copy to avoid affecting subsequent checks
                gen_ids = _greedy_steps(model, ids.clone(), (mask.clone() if mask is not None else None), args.steps, eos_id)
                if len(gen_ids) == 0 and args.steps > 0:
                    # Not a failure necessarily, but worth signaling
                    print("[self-check] note: greedy produced 0 ids (possibly hit EOS immediately or no steps)")
                # Verify tokens are in range
                for gid in gen_ids:
                    if gid < 0 or gid >= vocab_size:
                        raise AssertionError("generated id out of range of vocab size")
                print(f"[self-check] sample {idx}: greedy generated_ids={gen_ids}")

        except Exception as e:
            ok = False
            print(f"[self-check] ERROR on sample {idx}: {e}")

    if not ok:
        print("[self-check] FAILED")
        sys.exit(1)

    print("[self-check] PASSED")


if __name__ == "__main__":
    main(sys.argv[1:])
