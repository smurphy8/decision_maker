# DecisionMaker

A PyTorch wrapper for Hugging Face causal language models with TorchScript support.

## Overview

The `DecisionMaker` class provides a thin orchestration layer around Hugging Face's `AutoModelForCausalLM` instances. It exposes a TorchScript-friendly `forward` method that operates on tensors, while maintaining higher-level Python helpers for prompt-based generation.

## Features

- 🚀 **TorchScript Compatible**: Export models for production deployment via tracing
- 🤖 **Simple API**: High-level interface for text generation from prompts
- ⚡ **Flexible Generation**: Configurable temperature, top-p sampling, and token limits
- 🔧 **Device Management**: Easy GPU/CPU switching with dtype support
- 📦 **Minimal Dependencies**: Built on PyTorch and Transformers

## Installation

```bash
pip install torch transformers
```

## Quick Start

```python
from decision_maker import DecisionMaker

# Initialize with a model from Hugging Face Hub
maker = DecisionMaker("gpt2")

# Generate text from a prompt
response = maker.generate_from_prompt("Decide: should we launch today?")
print(response.text)
```

## Usage Examples
### One liner
`$>  uv run main.py --export-torchscript decision_maker.pt --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

### Inferno Export CLI
The `decision_maker_inferno.py` script exports a forward-only TorchScript module tailored for InfernoML and can optionally run a quick forward test using a local tokenizer.

Basic export:

```bash
uv run decision_maker_inferno.py
```

Common options:

- `--model` (default: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`)
- `--device` (`cpu`, `cuda`, `cuda:0`)
- `--dtype` (`float32`, `float16`, `bfloat16`)
- `--sequence-length` (default: `8`)
- `--batch-size` (default: `1`)
- `--without-mask` (trace without `attention_mask`)
- `--output` (default: `decision_maker_inferno.pt`)
- `--export-tokenizer` (save a TorchScript tokenizer sidecar using local assets)
- `--test-forward` (run a quick post-export forward)
- `--tokenizer` (local tokenizer path for the test; default: `tokenizer_config`)

Examples:

```bash
# Export with custom shapes
uv run decision_maker_inferno.py --sequence-length 16 --batch-size 1 --output decision_maker_inferno.pt

# Export and verify with a quick forward test
uv run decision_maker_inferno.py --test-forward

# Export without an attention mask
uv run decision_maker_inferno.py --without-mask

# Use specific device and dtype
uv run decision_maker_inferno.py --device cuda:0 --dtype float16 --test-forward

# Export model and a TorchScript tokenizer sidecar
uv run decision_maker_inferno.py --export-tokenizer tokenizer_ts.pt
```

### Basic Generation

```python
maker = DecisionMaker(
    "gpt2",
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.95
)

result = maker.generate_from_prompt("The future of AI is")
print(f"Generated: {result.text}")
print(f"Tokens generated: {result.generated_tokens.shape}")
```

### Custom Generation Parameters

```python
# Override defaults for a specific generation
result = maker.generate_from_prompt(
    "Write a haiku about Python:",
    max_new_tokens=100,
    temperature=0.9,
    top_p=0.98,
    return_full_text=True  # Include prompt in output
)
```

### GPU Acceleration

```python
# Use GPU with half precision
maker = DecisionMaker(
    "gpt2",
    device="cuda",
    torch_dtype="float16"
)

# Move to different device
maker.to_device("cpu")
```

### TorchScript Export

```python
# Export for production deployment
scripted_model = maker.as_torchscript(
    sequence_length=128,
    batch_size=4,
    use_attention_mask=True
)

# Save the scripted model
torch.jit.save(scripted_model, "decision_maker.pt")

# Load and use
loaded = torch.jit.load("decision_maker.pt")
output = loaded(input_ids, attention_mask)
```

### Tokenizer Trace

Print a structured tokenizer trace (ids, tokens, and offsets when available):

```bash
uv run main.py --model gpt2 --prompt "Hello, world" --trace-tokenizer
```

From Python, you can access the same information:

```python
maker = DecisionMaker("gpt2")
trace = maker.tokenizer_trace("Hello, world")
print(trace.input_ids)
print(trace.tokens)
print(trace.offsets)  # may be None when offsets are unsupported
```

### TracerWarnings Explained

### TorchScript Tokenizer (Experimental)

You can export a TorchScript-compatible ByteLevel BPE tokenizer that consumes raw bytes and returns token ids. This is useful when your deployment stack only supports TorchScript modules.

Export and smoke test:

```bash
uv run tokenizer_ts.py --test --output tokenizer_ts.pt
```

Use from Python:

```python
import torch

# Load scripted tokenizer
tok = torch.jit.load("tokenizer_ts.pt")

# Encode a UTF-8 string as bytes and run the tokenizer
prompt = "Hello from TorchScript tokenizer!\n"
inp = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long)
input_ids, attention_mask = tok(inp, max_length=128, add_eos=True)
```

Notes:
- This implementation is ByteLevel BPE and uses the local `tokenizer_config/tokenizer.json` vocabulary and merges.
- It segmentizes by whitespace and applies BPE merges; for typical ASCII text it aligns well with HF fast tokenizers, though it does not replicate the exact regex-based pretokenizer, so certain edge cases can differ.
- Input must be a 1D tensor of byte values; convert your prompt to UTF-8 bytes in your host environment before calling.
When exporting via tracing, you may see warnings like:

- `TracerWarning: Converting a tensor to a Python boolean/list ...`
- `torch.tensor results are registered as constants in the trace ...`

These arise from a tracing-friendly attention-mask shim active during export and are expected. They are safe to ignore for export purposes but indicate the trace is specialized to the shapes used during tracing (e.g., `sequence_length=8`, `batch_size=1`). If you need different shapes at inference time, re-export with the desired shapes.

## API Reference

### DecisionMaker Class

#### `__init__(model_name, *, max_new_tokens=128, temperature=0.7, top_p=0.95, device=None, torch_dtype=None)`

Initialize a new DecisionMaker instance.

**Parameters:**
- `model_name` (str): Hugging Face model identifier
- `max_new_tokens` (int): Default maximum tokens to generate
- `temperature` (float): Sampling temperature (0 = greedy)
- `top_p` (float): Nucleus sampling probability threshold
- `device` (str): Target device ("cuda", "cpu", or None for auto-detect)
- `torch_dtype` (str): Model precision ("float32", "float16", "bfloat16")

#### `generate_from_prompt(prompt, *, max_new_tokens=None, temperature=None, top_p=None, return_full_text=False)`

Generate text from a prompt string.

**Returns:** `GenerationResult` with:
- `text`: Generated text
- `generated_tokens`: Raw token tensor
- `prompt_length`: Number of prompt tokens

#### `forward(input_ids, attention_mask=None)`

TorchScript-compatible forward pass returning logits for the final position.

#### `as_torchscript(*, sequence_length=8, batch_size=1, use_attention_mask=True)`

Export the model as a TorchScript module via tracing.

## Supported Data Types

The following dtype strings are supported:
- `"float32"`, `"float"`, `"fp32"` → torch.float32
- `"float16"`, `"half"`, `"fp16"` → torch.float16
- `"bfloat16"`, `"bf16"` → torch.bfloat16

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+

## Advanced Features

### Attention Mask Patching

The module includes a context manager (`_torchscript_mask_patch`) that replaces SDPA mask generation with a tracing-friendly implementation, ensuring compatibility with TorchScript export.

### Automatic Tokenizer Configuration

The module automatically handles missing pad tokens by reusing the EOS token when necessary, ensuring compatibility with various model configurations.
