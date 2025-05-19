import torch
from safetensors.torch import load_file
import sys
import os

def convert_safetensors_to_pt(input_path, output_path=None):
    if not input_path.endswith(".safetensors"):
        raise ValueError("输入文件必须是 .safetensors 格式")

    print(f"Loading: {input_path}")
    state_dict = load_file(input_path)

    if output_path is None:
        output_path = input_path.replace(".safetensors", ".pt")

    torch.save(state_dict, output_path)
    print(f"Saved .pt to: {output_path}")

if __name__ == "__main__":
    input_file = "your_model.safetensors"  
    convert_safetensors_to_pt(input_file)
