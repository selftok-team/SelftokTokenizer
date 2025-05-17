# Copyright (C) 2025. Huawei Technologies Co., Ltd.  All rights reserved.

# Licensed under MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import sys
import os
print(sys.path)
sys.path.append(".")

import torch

# ####For running on GPU, comment out this part
# import torch_npu
# torch_npu.npu.set_compile_mode(jit_compile=False)
# from torch_npu.contrib import transfer_to_npu
# #####

from torchvision import transforms
import numpy as np
import argparse
import sys
from PIL import ImageFile
from copy import deepcopy
from collections import OrderedDict
from mimogpt.models.selftok.image_tokenizer import ImageTokenizer
ImageFile.LOAD_TRUNCATED_IMAGES = True
from mimogpt.models.selftok.sd3.sd3_impls import SDVAE, SD3LatentFormat
from mimogpt.models.selftok.sd3.rectified_flow import RectifiedFlow
from mimogpt.utils import hf_logger
from torchvision.utils import save_image


def load_state(model, state_dict, prefix='',init_method = None):
    model_dict = model.state_dict()  # 当前网络结构
    if prefix == 'model.diffusion_model.':
        excluded_keys = ['context_embedder.bias', 'context_embedder.weight']
        if init_method == 1:
            excluded_keys = ['context_embedder.bias', 'context_embedder.weight', 'final_layer.adaLN_modulation.1.bias', 'final_layer.adaLN_modulation.1.weight', 'final_layer.linear.bias', 'final_layer.linear.weight']
            pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict and k.replace(prefix,'') not in excluded_keys and 'context_block' not in k}
        elif init_method == 2:
            pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict and k.replace(prefix,'') not in excluded_keys and 'context_block' not in k and 'x_block.attn' not in k}
        else:
            pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict and k.replace(prefix,'') not in excluded_keys and 'context_block' not in k}
    else:
        pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict}
    dict_t = deepcopy(pretrained_dict)
    for key, weight in dict_t.items():
        if key in model_dict and model_dict[key].shape != dict_t[key].shape:
            pretrained_dict.pop(key)
   
    m, u = model.load_state_dict(pretrained_dict, strict=False)
    if len(m) > 0:
        hf_logger.info(f"model missing keys:{m}")
    if len(u) > 0:
        hf_logger.info(f"mode unexpected keys:{u}")

class NormalizeToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, reshape=True):
        self.reshape = reshape

    def __call__(self, image):
        image = np.array(image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        if self.reshape:
            image = np.reshape(image, (image.shape[0], image.shape[1], -1))
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def set_sd3_vae(vae_path, device):
    vae = SDVAE(device="cpu", dtype=torch.bfloat16)
    state_dict = torch.load(vae_path, map_location='cpu')
    load_state(vae, state_dict, 'first_stage_model.')
    vae.to(device)
    vae.eval()
    return vae

def set_ema_model(model, device):
    ema = deepcopy(model).to(torch.float32)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)
    ema = ema.to(device)
    ema.eval()
    return ema

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

def str_to_bool(value):
    """Converts a string to a boolean.

    Raises argparse.ArgumentTypeError if the value cannot be parsed as a boolean.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class SelftokPipeline():
    def __init__(self, cfg, ckpt_path, datasize = 256, start = 1.0, cfg_scale = 1,model_type='sd3', dtype=torch.bfloat16, ema_decoder=False, device=None):
        
        self.cfg = cfg
        self.datasize = datasize
        self.model_type = model_type
        self.dtype = dtype
        # define models
        if self.model_type == 'sd3':
            self.vae = set_sd3_vae(cfg.common.vae_path, device)
        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {self.model_type}. Expected 'sd3'")
        cfg.tokenizer.params.noise_schedule_config.is_eval=cfg.common.is_eval
        
        self.model = ImageTokenizer(**cfg.tokenizer.params)
        self.model.set_eval()

        self.ema_decoder = ema_decoder
        if self.ema_decoder==True:
            self.ema = set_ema_model(self.model.model, device)
            self.ema.eval()
        self.vae.eval()
        self.diti = self.model.diti
        self.K = self.diti.K
        self.count = 0
        self.count_cfg = 0
        self.start = start
        self.cfg_scale = cfg_scale
        if hasattr(cfg.tokenizer.params, "cut_of_k") and self.cfg.tokenizer.params.cut_of_k:
            self.cut_of_k = self.cfg.tokenizer.params.cut_of_k
        else:
            self.cut_of_k = None
            
       
        pretrain = ckpt_path
        
        state_dict = torch.load(pretrain, map_location="cpu")
        print(f"Loading all...")
        if self.ema_decoder==True:
            self.ema.load_state_dict(state_dict['ema_state_dict'])
        self.model.load_state_dict(state_dict['state_dict'], strict=False)
        
        if self.ema_decoder==True:
            self.ema.to(device)
        self.model.to(device)
        
        self._steps = 50
        self.flow = RectifiedFlow(
            self._steps, self.start, self.cut_of_k, val_schedule='uniform', shift=1.0, **cfg.tokenizer.params.noise_schedule_config,
        )

        self.cond_vary = True
        self.saved_images = 8


    def encoding(self, images, device):

        print(f"Begin encoding.")

        images = images.to(dtype=self.dtype, device=device) # Move images to GPU once per batch
        x_0 = self.vae.encode(images)
        x_0 = SD3LatentFormat().process_in(x_0)
        
        x_0 = x_0.to(torch.float32)

        with torch.no_grad():
            _, tokens = self.model.encoder(x_0, d=None)

        print('End encoding.')
        
        return tokens

    @torch.no_grad()
    def decoding(self, idx, device):

        print(f"Begin decoding.")
        
        token_idx = torch.from_numpy(idx).to(device)
        B = token_idx.shape[0]

        
        outs_q = self.model.encoder.quantizer.get_output_from_indices(token_idx)
        outs_q = outs_q.reshape(B, -1, outs_q.shape[-1])

        if self.model.encoder.post_norm:
            outs_q = self.model.encoder.final_layer_norm3(outs_q)
        
        if hasattr(self.cfg.tokenizer.params, "stages"):
            t_mapped = torch.tensor([self.flow.timestep_map[0]]*B, device=device).long()
        else:
            t_mapped = torch.tensor([(self.flow.timestep_map[0])/1000.0]*B, device=device)
        k = self.diti.to_indices(t_mapped)
        d=k
        
        enc_mask = self.model.encoder.get_encoder_mask(token_idx, d)
        attn_mask = enc_mask
        mask_v = enc_mask[..., None].expand_as(outs_q)
        encoder_hidden_states = outs_q.cuda() * mask_v.cuda()
        
        ori_hidden_states = outs_q

        model_kwargs = dict(
            encoder_hidden_states=encoder_hidden_states,
            mask=attn_mask,
            context_see_xt=True
        )
        
        latent_dim = self.datasize // 8

        xt = torch.randn(B, 16, latent_dim, latent_dim).to(device)

        kwargs = {}
        enc_in =xt.float()

        if self.ema_decoder==True:
            pred_x0 = self.flow.p_sample_loop(
                self.ema.model, xt.shape, xt, model_kwargs=model_kwargs,
                start_t=self._steps, cond_vary=self.cond_vary,
                diti=self.diti, encoder=self.model.encoder, x_0=enc_in,
                ori_hidden_states=ori_hidden_states,**kwargs
            )
        else: # here
            pred_x0 = self.flow.p_sample_loop(
                self.model.model, xt.shape, xt, model_kwargs=model_kwargs,
                start_t=self._steps, cond_vary=self.cond_vary,
                diti=self.diti, encoder=self.model.encoder, x_0=enc_in,
                ori_hidden_states=ori_hidden_states,**kwargs
            )

        if self.model_type == 'sd3':
            pred_x0_out = SD3LatentFormat().process_out(pred_x0)

            pred_x0_out = pred_x0_out.to(self.dtype)
            recons = self.vae.decode(pred_x0_out)
        
        norm_ip(recons, -1, 1)

        print('End decoding.')
        
        return recons
    
    @torch.no_grad()
    def decoding_with_renderer(self, idx, device):
        
        print(f"Begin decoding with Renderer.")
        
        token_idx = torch.from_numpy(idx).to(device)
        B = token_idx.shape[0]

        outs_q = self.model.encoder.quantizer.get_output_from_indices(token_idx)
        outs_q = outs_q.reshape(B, -1, outs_q.shape[-1])

        if self.model.encoder.post_norm:
            outs_q = self.model.encoder.final_layer_norm3(outs_q)
        
        pred_x0, _ = self.model.model(y=None, encoder_hidden_states=outs_q)
        
        if self.model_type == 'sd3':
            pred_x0_out = SD3LatentFormat().process_out(pred_x0)
            
            pred_x0_out = pred_x0_out.to(self.dtype)
            recons = self.vae.decode(pred_x0_out)
        
        norm_ip(recons, -1, 1)

        print('End decoding with Renderer.')
        
        return recons



    