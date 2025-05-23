import os
import sys
print(sys.path)
sys.path.append(".")

import argparse
from mimogpt.infer.infer_utils import parse_args_from_yaml
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from mimogpt.infer.SelftokPipeline_gpu import SelftokPipeline
from mimogpt.infer.SelftokPipeline_gpu import NormalizeToTensor
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--yml-path", type=str, default="./configs/res256/256-eval.yml") # download from https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium.safetensors?download=true, require huggingface login, you have to change the format to .pt with safetensor_to_pt.py
parser.add_argument("--pretrained", type=str, default="path/to/your/ckpt.pth") 
parser.add_argument("--data_size", type=int, default=512)

args = parser.parse_args()

cfg = parse_args_from_yaml(args.yml_path)
model = SelftokPipeline(cfg=cfg, ckpt_path=args.pretrained, datasize=args.data_size, device='cuda')

img_transform = transforms.Compose([
    transforms.Resize(args.data_size),
    transforms.CenterCrop(args.data_size),
    NormalizeToTensor(),
])

image_paths = ['img_1.png', 'img_2.png']
images = [img_transform(Image.open(p)) for p in image_paths]
images = torch.stack(images).to('cuda')

tokens = model.encoding(images, device='cuda')
np.save('./token.npy', tokens.detach().cpu().numpy())


tokens = np.load('./token.npy')
    
images = model.decoding(tokens, device='cuda')
for b in range(len(images)):
    save_image(images[b], f"./re_{b}_{args.data_size}_2.png")

# images = model.decoding_with_renderer(tokens, device='cuda')
# for b in range(len(images)):
#     save_image(images[b], f"./re_renderer_{b}_{args.data_size}_2.png")