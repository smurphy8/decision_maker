"""Utilities for wrapping a Hugging Face causal language model behind a simple API.

The :class:`DecisionMaker` class is intended to be a thin orchestrator around an
AutoModelForCausalLM instance.  It exposes a torchscript-friendly ``forward``
method that operates on tensors, while higher level helpers such as
``generate_from_prompt`` remain pure Python and are therefore excluded from
torchscript through ``torch.jit.ignore``.

Typical usage::

    maker = DecisionMaker("sshleifer/tiny-gpt2")
    response = maker.generate_from_prompt("Decide: do we launch today?")
    print(response.text)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import contextlib
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)


@dataclass
class GenerationResult:
    """Container for decoded model output."""

    text: str
    generated_tokens: torch.Tensor
    prompt_length: int


class DecisionMaker(nn.Module):
    """Wrapper around a Hugging Face causal LM with single-prompt generation."""

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = self._resolve_dtype(torch_dtype)

        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._init_model_and_tokenizer()
        self._device_logged = False
        self._log_device_state()

        # Register a dummy buffer so ``to()`` works as expected when scripting.
        self.register_buffer("_device_buffer", torch.zeros(1), persistent=False)

    @staticmethod
    def _resolve_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
        if dtype_name is None:
            return None
        normalized = dtype_name.lower()
        mapping: Dict[str, torch.dtype] = {
            "float32": torch.float32,
            "float": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "half": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported dtype string: {dtype_name}")
        return mapping[normalized]

    @torch.jit.ignore
    def _init_model_and_tokenizer(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token_id is None:
            # Hugging Face generation requires a pad token; reuse eos if absent.
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer

        config = AutoConfig.from_pretrained(self.model_name)
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "eager"
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self._dtype,
            config=config,
        )
        self.model.to(self._device_str)
        self.model.eval()

    @torch.jit.ignore
    def _log_device_state(self) -> None:
        if getattr(self, "_device_logged", False):
            return
        param = next(self.model.parameters(), None)
        model_device = param.device if param is not None else torch.device(self._device_str)
        model_dtype = param.dtype if param is not None else (self._dtype or torch.float32)
        print(
            f"[DecisionMaker] Model '{self.model_name}' initialized on {model_device} (dtype={model_dtype})"
        )
        self._device_logged = True

    @torch.jit.ignore
    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer has not been initialized.")
        return self._tokenizer

    @torch.jit.ignore
    def device(self) -> torch.device:
        return torch.device(self._device_str)

    @torch.jit.export
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the underlying model and return logits for the final position."""
        if attention_mask is None:
            outputs = self.model(input_ids=input_ids)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits[:, -1, :]

    @torch.jit.ignore
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate continuations directly from input tensors."""
        max_tokens = max_new_tokens or self.max_new_tokens
        use_temperature = temperature if temperature is not None else self.temperature
        use_top_p = top_p if top_p is not None else self.top_p
        do_sample = use_temperature > 0

        gen_kwargs: Dict[str, float | int | bool] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_p": use_top_p,
            "pad_token_id": self.get_tokenizer().pad_token_id,
            "eos_token_id": self.get_tokenizer().eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = use_temperature
        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask

        with torch.no_grad():
            generated = self.model.generate(**inputs, **gen_kwargs)
        return generated

    @torch.jit.ignore
    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        return_full_text: bool = False,
    ) -> GenerationResult:
        """Encode a single prompt string, generate tokens, and decode the output."""
        tokenizer = self.get_tokenizer()
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device())
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device())
        print(f"[DecisionMaker] Prompt tensors placed on device {input_ids.device}")

        generated = self.generate(
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        prompt_length = input_ids.size(1)
        decoded = tokenizer.decode(
            generated[0] if return_full_text else generated[0, prompt_length:],
            skip_special_tokens=True,
        )
        return GenerationResult(
            text=decoded,
            generated_tokens=generated.cpu(),
            prompt_length=prompt_length,
        )

    @torch.jit.ignore
    def to_device(self, device: str | torch.device) -> "DecisionMaker":
        """Convenience helper to move both model and metadata to a device."""
        self._device_str = str(device)
        self.model.to(device)
        self._device_buffer = self._device_buffer.to(device)
        self._device_logged = False
        self._log_device_state()
        return self

    @torch.jit.ignore
    def as_torchscript(
        self,
        *,
        sequence_length: int = 8,
        batch_size: int = 1,
        use_attention_mask: bool = True,
    ) -> torch.jit.ScriptModule:
        """Return a torchscript version of the module via tracing."""
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        tokenizer = self.get_tokenizer()
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError("Tokenizer must define pad_token_id before tracing.")
        eos_id = tokenizer.eos_token_id or pad_id

        device = self.device()
        input_ids = torch.full(
            (batch_size, sequence_length),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        input_ids[:, -1] = eos_id

        trace_inputs: Dict[str, Tuple[torch.Tensor, ...]]
        if use_attention_mask:
            attention_mask = torch.ones_like(input_ids)
            trace_inputs = {"forward": (input_ids, attention_mask)}
        else:
            trace_inputs = {"forward": (input_ids,)}

        wrapper = _ForwardOnlyWrapper(self.model)
        wrapper.to(device)
        wrapper.eval()

        with _torchscript_mask_patch():
            traced = torch.jit.trace_module(wrapper, trace_inputs, strict=False)
        return traced


@contextlib.contextmanager
def _torchscript_mask_patch():
    """Temporarily replace the SDPA mask helper with a tracing-friendly variant."""
    import transformers.masking_utils as masking_utils

    original_sdpa_mask = masking_utils.sdpa_mask
    original_sdpa_registered = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.get("sdpa")

    def _slow_sdpa_mask(
        batch_size: int,
        cache_position: torch.Tensor,
        kv_length: int,
        kv_offset: int = 0,
        mask_function=masking_utils.causal_mask_function,
        attention_mask: Optional[torch.Tensor] = None,
        local_size: Optional[int] = None,
        allow_is_causal_skip: bool = True,
        allow_torch_fix: bool = True,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        q_length = cache_position.shape[0]
        padding_mask = masking_utils.prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

        if allow_is_causal_skip and masking_utils._ignore_causal_mask_sdpa(
            padding_mask, q_length, kv_length, kv_offset, local_size
        ):
            return None

        mask_fn = mask_function
        if padding_mask is not None:
            mask_fn = masking_utils.and_masks(mask_fn, masking_utils.padding_mask_function(padding_mask))

        cache_positions = [int(pos) for pos in cache_position.detach().cpu().tolist()]
        kv_positions = list(range(kv_offset, kv_offset + kv_length))

        mask = torch.zeros((batch_size, 1, q_length, kv_length), dtype=torch.bool, device=cache_position.device)
        head_idx_tensor = cache_position.new_tensor(0)
        for batch_idx in range(batch_size):
            for q_idx, q_pos in enumerate(cache_positions):
                q_tensor = cache_position.new_tensor(q_pos)
                for kv_rel_idx, kv_pos in enumerate(kv_positions):
                    kv_tensor = cache_position.new_tensor(kv_pos)
                    value = mask_fn(batch_idx, head_idx_tensor, q_tensor, kv_tensor)
                    if isinstance(value, torch.Tensor):
                        value = bool(value.detach().cpu().item())
                    else:
                        value = bool(value)
                    mask[batch_idx, 0, q_idx, kv_rel_idx] = value

        if allow_torch_fix:
            mask |= torch.all(~mask, dim=-1, keepdim=True)
        return mask

    masking_utils.sdpa_mask = _slow_sdpa_mask
    if original_sdpa_registered is not None:
        masking_utils.ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", _slow_sdpa_mask)

    try:
        yield
    finally:
        masking_utils.sdpa_mask = original_sdpa_mask
        if original_sdpa_registered is not None:
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", original_sdpa_registered)


class _ForwardOnlyWrapper(nn.Module):
    """Thin wrapper to expose DecisionMaker.forward for tracing."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            outputs = self.model(input_ids=input_ids)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits[:, -1, :]
