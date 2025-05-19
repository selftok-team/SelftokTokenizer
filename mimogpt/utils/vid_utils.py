# -*- coding: utf-8 -*-

import os
import math
import torch
import numpy as np
from PIL import Image

try:
    from decord import cpu, gpu
    from decord import VideoReader
except:
    print("# Ascend didn't support decord, skipped")


def load_image(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def image_to_tensor(image_file, preprocess):
    image_data = preprocess(load_image(image_file))
    return image_data


def video_to_tensor_decord(video_file, preprocess, max_frames=8, start=None, end=None):
    vr = VideoReader(video_file, ctx=cpu(0))
    fps = vr.get_avg_fps()

    if start is None or start == 0:
        start = 0
    else:
        start = max(math.floor(start * fps), 0)

    if end is None or end == -1:
        end = len(vr)
    else:
        end = min(math.ceil(end * fps), len(vr))
    assert start < end, "vid name: {}, start: {} and end: {}".format(os.path.basename(video_file), start, end)

    idx_list = np.linspace(start, end - 1, max_frames, dtype="int").tolist()
    frames = vr.get_batch(idx_list).asnumpy()
    video_data = [preprocess(Image.fromarray(img, mode="RGB")) for img in frames]
    return torch.tensor(np.stack(video_data))
