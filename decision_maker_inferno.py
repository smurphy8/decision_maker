"""
A version of DecisionMaker that is designed for the onping the InfernoMl framework.
This requires a wrapper around the original DecisionMaker class to adapt its interface.
for training.

"""
import torch
import torch.export
from torch import nn
from decision_maker import DecisionMaker
from torch.export import export, ExportedProgram

class InfernoDecisionWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = DecisionMaker(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            max_new_tokens=256,
            temperature=0.9,
        )

    def forward(self, x):
        return self.model.generate_from_prompt(
            prompt=(
                "Answer this Question: what is the capital of france?\n"
                "Just give me the answer in one word.\n"
            ),
            max_new_tokens=180,
        )


def main():    # Example usage of DecisionMaker within the InfernoMl framework 
    mp = InfernoDecisionWrapper()
    rslt = mp.forward(torch.tensor([1, 2]))
    print(rslt)

if __name__ == "__main__":
    main()
