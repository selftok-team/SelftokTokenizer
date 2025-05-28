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
from mimogpt.infer.SelftokPipeline import SelftokPipeline
from mimogpt.infer.SelftokPipeline import NormalizeToTensor
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--yml-path", type=str, default="./configs/res256/256-eval.yml") # download from https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium.safetensors?download=true, require huggingface login, you have to change the format to .pt with safetensor_to_pt.py
parser.add_argument("--pretrained", type=str, default="/home/wwj/SelftokTokenizer/weight/SelftokTokenizer/tokenizer_512_ckpt.pth") 
parser.add_argument("--sd3_pretrained", type=str, default="/home/wwj/SelftokTokenizer/weight/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671") 
parser.add_argument("--data_size", type=int, default=256)

args = parser.parse_args()

cfg = parse_args_from_yaml(args.yml_path)
model = SelftokPipeline(cfg=cfg, ckpt_path=args.pretrained, sd3_path=args.sd3_pretrained, datasize=args.data_size, device='cuda')

img_transform = transforms.Compose([
    transforms.Resize(args.data_size),
    transforms.CenterCrop(args.data_size),
    NormalizeToTensor(),
])

image_paths = ['./test.jpg']
images = [img_transform(Image.open(p)) for p in image_paths]
images = torch.stack(images).to('cuda')

tokens = model.encoding(images, device='cuda')
np.save('./token.npy', tokens.detach().cpu().numpy())
tokens = np.load('./token.npy')
    
images = model.decoding(tokens, device='cuda')
for b in range(len(images)):
    save_image(images[b], f"./re_{b}_{args.data_size}_2.png")
