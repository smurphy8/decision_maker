Make a Readme


I'll create a README file for this `decision_maker.py` module that explains its purpose, features, and usage.Now I'll write a comprehensive README for the DecisionMaker module:

```markdown
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

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]
```

This README provides:
- Clear overview of the module's purpose
- Key features and benefits
- Installation instructions
- Quick start example
- Comprehensive usage examples
- Full API documentation
- Technical details about TorchScript support
- Requirements and compatibility information

The README is structured to help users quickly understand what the DecisionMaker class does and how to use it effectively.
