from __future__ import annotations

"""
TorchScript-friendly ByteLevel BPE tokenizer.

This module provides a scriptable tokenizer that consumes raw bytes as a
torch.LongTensor (values 0..255) and produces token ids using a ByteLevel BPE
vocabulary and merges. It avoids Python-side Hugging Face tokenizers and can be
saved as a TorchScript artifact.

Notes and caveats:
- This aims for practical compatibility with ByteLevel BPE tokenizers
  (GPT-2/Qwen2-style). It handles whitespace as ByteLevel byte-to-unicode
  markers and applies BPE merges. While reasonably faithful, it does not
  replicate the exact pre-tokenizer regex used by Hugging Face; segmentation
  is whitespace-driven. In most ASCII text it will align, but edge cases may
  diverge from HF fast tokenizers.
- Input must be bytes (a 1D tensor of values in [0,255]). If you only have a
  Python string on the host side, encode it to UTF-8 and pass the bytes.
"""

from typing import Dict, List, Tuple, Optional

import torch
from torch import nn


def _bytes_to_unicode() -> List[str]:
    """Return the canonical GPT-2 byte->unicode mapping as a list[str] of len 256.

    Mirrors the standard mapping used by GPT-2's ByteLevel BPE, where printable
    bytes map to themselves and other bytes map to higher unicode code points.
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


class ScriptedByteLevelBPETokenizer(nn.Module):
    """Scriptable ByteLevel BPE tokenizer.

    Constructor expects already-loaded vocab and merges (from tokenizer.json),
    and special token ids.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        pad_id: Optional[int],
        eos_id: Optional[int],
    ) -> None:
        super().__init__()
        self.vocab = vocab
        # Encode merge pairs as single-string keys using a sentinel delimiter.
        self._delim = "\u0001"
        bpe_ranks: Dict[str, int] = {}
        for i, (l, r) in enumerate(merges):
            bpe_ranks[l + self._delim + r] = i
        self.bpe_ranks = bpe_ranks

        # Byte-level mapping and whitespace markers
        self.byte_map = _bytes_to_unicode()
        # Whitespace bytes to segment on (tab, LF, VT, FF, CR, space)
        whitespace_bytes = [9, 10, 11, 12, 13, 32]
        ws: Dict[str, int] = {}
        for b in whitespace_bytes:
            ws[self.byte_map[b]] = 1
        self.ws_markers = ws

        self.pad_id = -1 if pad_id is None else int(pad_id)
        self.eos_id = -1 if eos_id is None else int(eos_id)

    def _is_ws_marker(self, ch: str) -> bool:
        return ch in self.ws_markers

    def _split_segments(self, mapped: str) -> List[str]:
        """Split a byte-level mapped string into segments using whitespace markers.

        Whitespace markers are absorbed as a prefix into the next segment, which
        reproduces the common "Ġ"-prefixed word behavior for BPE.
        """
        segs: List[str] = []
        cur = ""
        ws_prefix = ""
        for i in range(len(mapped)):
            ch = mapped[i]
            if self._is_ws_marker(ch):
                ws_prefix += ch
                if cur != "":
                    segs.append(cur)
                    cur = ""
                continue
            if cur == "":
                cur = ws_prefix + ch
                ws_prefix = ""
            else:
                cur += ch
        if cur != "":
            segs.append(cur)
        return segs

    def _bpe(self, token: str) -> List[str]:
        """Apply BPE merges to a single token string and return final symbols."""
        if len(token) <= 1:
            return [token]
        # Initialize as a list of characters
        word: List[str] = [token[i] for i in range(len(token))]
        if len(word) == 1:
            return [token]
        while True:
            best_rank = 2147483647
            best_l = ""
            best_r = ""
            # Find best-ranked adjacent pair
            for i in range(len(word) - 1):
                l = word[i]
                r = word[i + 1]
                key = l + self._delim + r
                rank = self.bpe_ranks.get(key)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_l = l
                    best_r = r
            if best_rank == 2147483647:
                break
            # Merge all occurrences of best pair
            merged: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_l and word[i + 1] == best_r:
                    merged.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
            if len(word) == 1:
                break
        return word

    def _token_to_ids(self, tok: str) -> List[int]:
        """Convert a merged token string to vocab ids, with char fallback."""
        out: List[int] = []
        if tok in self.vocab:
            out.append(self.vocab[tok])
            return out
        # Fallback to char-level decomposition
        for i in range(len(tok)):
            ch = tok[i]
            if ch in self.vocab:
                out.append(self.vocab[ch])
        return out

    def forward(
        self,
        input_bytes: torch.Tensor,
        max_length: int = 2048,
        add_eos: bool = True,
    ):
        """Encode bytes to token ids.

        Args:
            input_bytes: 1D Long tensor of values in [0,255]
            max_length: maximum sequence length after optional EOS; remaining is padded
            add_eos: append EOS if available and space permits

        Returns:
            (input_ids, attention_mask) as 1D Long tensors of shape [max_length]
        """
        if input_bytes.dim() != 1:
            raise RuntimeError("input_bytes must be a 1D tensor of byte values")
        n = int(input_bytes.size(0))
        # Build mapped unicode string from byte values
        s = ""
        for i in range(n):
            b = int(input_bytes[i].item())
            if b < 0:
                b = 0
            if b > 255:
                b = 255
            s += self.byte_map[b]

        # Segment and BPE-encode
        segs = self._split_segments(s)
        ids: List[int] = []
        for seg in segs:
            pieces = self._bpe(seg)
            for p in pieces:
                ids.extend(self._token_to_ids(p))

        # Add EOS if requested and available
        if add_eos and self.eos_id >= 0:
            ids.append(self.eos_id)

        # Truncate and pad
        if max_length <= 0:
            raise RuntimeError("max_length must be positive")
        if len(ids) > max_length:
            ids = ids[:max_length]

        pad_id = self.pad_id if self.pad_id >= 0 else (self.eos_id if self.eos_id >= 0 else 0)
        attn: List[int] = [1 for _ in range(len(ids))]
        while len(ids) < max_length:
            ids.append(pad_id)
            attn.append(0)

        device = input_bytes.device
        return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(attn, dtype=torch.long, device=device)


def load_tokenizer_assets(tokenizer_json_path: str) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """Load vocab and merges from a Hugging Face tokenizer.json."""
    import json

    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = data["model"]
    if model["type"].lower() != "bpe":
        raise ValueError(f"Unsupported tokenizer model type: {model['type']}")
    vocab: Dict[str, int] = {k: int(v) for k, v in model["vocab"].items()}
    merges_list: List[str] = model.get("merges", [])
    merges: List[Tuple[str, str]] = []
    for m in merges_list:
        parts = m.split(" ")
        if len(parts) == 2:
            merges.append((parts[0], parts[1]))
    return vocab, merges


def resolve_special_ids(
    vocab: Dict[str, int],
    tokenizer_config_path: Optional[str] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Resolve pad/eos ids from tokenizer_config.json when available."""
    pad_id: Optional[int] = None
    eos_id: Optional[int] = None
    if tokenizer_config_path is None:
        return pad_id, eos_id
    import json, os

    cfg_file = os.path.join(tokenizer_config_path, "tokenizer_config.json")
    if not os.path.exists(cfg_file):
        return pad_id, eos_id
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        pad_tok = cfg.get("pad_token")
        eos_tok = cfg.get("eos_token")
        if isinstance(pad_tok, str) and pad_tok in vocab:
            pad_id = int(vocab[pad_tok])
        if isinstance(eos_tok, str) and eos_tok in vocab:
            eos_id = int(vocab[eos_tok])
    except Exception:
        pass
    return pad_id, eos_id


def build_scripted_tokenizer(
    tokenizer_dir: str = "tokenizer_config",
) -> torch.jit.ScriptModule:
    """Build and script a ByteLevel BPE tokenizer from local assets."""
    import os

    tok_json = os.path.join(tokenizer_dir, "tokenizer.json")
    vocab, merges = load_tokenizer_assets(tok_json)
    pad_id, eos_id = resolve_special_ids(vocab, tokenizer_dir)
    tok = ScriptedByteLevelBPETokenizer(vocab, merges, pad_id, eos_id)
    return torch.jit.script(tok)


def _cli():
    import argparse
    import os

    p = argparse.ArgumentParser(description="Export a TorchScript ByteLevel BPE tokenizer")
    p.add_argument("--tokenizer", default="tokenizer_config", help="Path to directory with tokenizer.json & tokenizer_config.json")
    p.add_argument("--output", default="tokenizer_ts.pt", help="Where to save the scripted tokenizer module")
    p.add_argument("--test", action="store_true", help="Run a quick encode test before saving")
    p.add_argument("--max-length", type=int, default=128, help="Max length for the test encode")
    args = p.parse_args()

    scripted = build_scripted_tokenizer(args.tokenizer)
    if args.test:
        # Quick smoke test with ASCII prompt
        prompt = "Hello from TorchScript tokenizer!\n"
        b = prompt.encode("utf-8")
        inp = torch.tensor(list(b), dtype=torch.long)
        with torch.no_grad():
            ids, attn = scripted(inp, max_length=args.max_length)
        print("ids shape:", tuple(ids.shape), "attn shape:", tuple(attn.shape))
        print("first 16 ids:", ids[:16].tolist())
    scripted.save(args.output)
    print(f"Saved TorchScript tokenizer to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    _cli()

