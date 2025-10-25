"""
A version of DecisionMaker that is designed for the InfernoMl framework.
This provides wrappers around the original DecisionMaker class to adapt its
interface for training and export scenarios.
"""
from typing import Optional, Tuple

import torch
from torch import nn

from decision_maker import DecisionMaker


class InfernoDecisionTokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        decision_maker = DecisionMaker(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            max_new_tokens=256,
            temperature=0.9,
        )
        # Maintain a direct reference for eager tokenization outside tracing.
        self._tokenizer = decision_maker.get_tokenizer()

    @torch.jit.ignore
    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize raw text into tensor inputs for the traced module."""
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask


class InfernoDecisionWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = DecisionMaker(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            max_new_tokens=256,
            temperature=0.9,
        )

    def forward(self, x) -> torch.Tensor:
        return self.model.generate_from_prompt(
            prompt=(
                "Answer this Question: what is the capital of france?\n"
                "Just give me the answer in one word.\n"
            ),
            max_new_tokens=180,
        )


def main() -> None:
    """Example usage of DecisionMaker within the InfernoMl framework."""
    tokenizer = InfernoDecisionTokenizer()
    tokenizer.eval()
    input_ids, attention_mask = tokenizer.encode("What is the capital of France?")
    traced = torch.jit.trace(tokenizer, (input_ids, attention_mask))
    traced.save("tokenizer.pt")
    print("Saved traced tokenizer module to tokenizer.pt")


if __name__ == "__main__":
    main()
