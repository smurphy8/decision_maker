"""
A version of DecisionMaker that is designed for the onping the InfernoMl framework.
This requires a wrapper around the original DecisionMaker class to adapt its interface.
for training.

"""
import torch
from decision_maker import DecisionMaker





def main():    # Example usage of DecisionMaker within the InfernoMl framework 


# [smurphy@rog-nixos:~/projects/inferno_ml_projects/decision_maker]$ uv run main.py --export-torchscript decision_maker.pt --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|xclip -selection clipboard
  maker = DecisionMaker(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        max_new_tokens=256,
                        temperature=0.9)
  

 


  result = maker.generate_from_prompt(
        prompt="""
        Answer this Question: what is the capital of france?
        Just give me the answer in one word.
        """,
        max_new_tokens=180
    )
  print(result.text)
  
if __name__ == "__main__":
    main()
