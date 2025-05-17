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



import os
import yaml
from collections import OrderedDict
import random
from .model_zoo import Enc_models, DiT_models
from .models_ours import Encoder
import torch
from mimogpt.utils import hf_logger
from torch import nn
import numpy as np
from diffusers.models import AutoencoderKL
from mimogpt.models.selftok.diffusion import create_diffusion
from mimogpt.models.selftok.diti_utils import DiTi
from mimogpt.models.selftok.sd3.rectified_flow import RectifiedFlow

MAX_LATENT_SIZE = 384


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class MultiImageTokenizer(nn.Module):
    def __init__(
        self,
        image_size,
        k,
        stages,
        k_per_stage,
        encoder_hidden_size,
        encoder_list,
        train_encoder_res,
        model,
        decoder_config,
        noise_schedule_config=None,
        gradient_checkpointing=False,
        in_channels=16,
        diffusion_type='flow',
        enable_enc_variable_size=False,    # to enable variable image size for encoder; max size=MAX_LATENT_SIZE after downsampling by vae
        **kwargs,
    ):
        super().__init__()

        # 253-272
        # Create model:
        # predict_xstart = False if init_with_pretrained else True
        train_filter = decoder_config['train_filter']
        freeze_filter = decoder_config['freeze_filter']
        decoder_config['train_filter'] = train_filter.split('+') if train_filter != 'all' else None
        decoder_config['freeze_filter'] = freeze_filter.split('+') if freeze_filter != '' else []
        self.diffusion_type = diffusion_type
        if diffusion_type == 'flow':
            self.diffusion = RectifiedFlow(**noise_schedule_config)
            self.recon_ratio = 1.0
        else:
            self.diffusion = create_diffusion(
                timestep_respacing="", predict_xstart=False, learn_sigma=True, use_kl=False
            )  # default: 1000 steps, linear noise schedule, pred xstart!

        assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        
        self.diti = DiTi(1000, k, stages, k_per_stage)

        latent_size = image_size // 8
        
        print('enable_enc_variable_size',enable_enc_variable_size)
        
                
        self.encoder = {}
        for _, info in encoder_list.__dict__.items():
            enc = info['enc']
            enc_res = info['enc_res']
            encoder_config = info['encoder_config']
            if 'Qformer' in enc:
                if enable_enc_variable_size:
                    encoder_config['pos_embed_max_size'] = MAX_LATENT_SIZE // int(enc[-1])
                    encoder_config['diti'] = self.diti
            else:
                if enable_enc_variable_size:
                    assert False, "Other encoder does not support variable input size."

            res_encoder = Enc_models[enc](
                K=k,
                input_size=latent_size,
                encoder_hidden_size=encoder_hidden_size,
                in_channels=in_channels,
                gradient_checkpointing=gradient_checkpointing,
                **encoder_config
            )
            self.encoder[enc_res] = res_encoder
            
        decoder_config['diti'] = self.diti
        self.model = DiT_models[model](
            K=k,
            input_size=latent_size, 
            encoder_hidden_size=encoder_hidden_size,
            in_channels=in_channels,
            gradient_checkpointing=gradient_checkpointing,
            **decoder_config
            )

        self.model.freeze()  # keep only params matching train_filter
        # recon prev timestep weight
        self.T = self.diffusion.num_timesteps
        self.train_encoder_res = train_encoder_res

    def set_train(self):
        self.model.train()
        self.encoder[self.train_encoder_res].train()
        
        # self.ema.eval()

    def set_eval(self):
        self.model.eval()
        for _, encoder in self.encoder.items():
            encoder.eval()
        # self.ema.eval()


    def forward(self, x, ema=None, full_tokens=False):
        if type(x).__name__== 'dict':
            device = x['img'].device
        else:
            
            device = x.device
        
        if self.diffusion_type == 'flow':
            if (x['img'][2] * x['img'][3] / 4096.0) < 0.5:
                shift = 1.0
                high_res = False
            else:
                shift = 1.818
                high_res = True
            t = self.diffusion.sample_t(x['img'][0],1.0).cuda()
            k_batch = self.diti.to_indices(t * 1000.0)
            if high_res:
                t = self.diffusion.shift_t(t, shift)
        else:
            t = torch.randint(0, self.diffusion.num_timesteps, (x['img'][0],), device=device)
            k_batch = self.diti.t_to_idx.to(device)[t]
        k_batch = self.diti.to_indices(torch.ones_like(t) * 1000.0) if full_tokens else k_batch
        
        # x的处理放在那儿
        
        # k-batch need change # 
        enc_h_list = []
        attn_mask_list = []
        quan_loss_list = []
        log_dict_list = []
        for res, encoder in self.encoder.items():
            if not encoder.training:
                with torch.no_grad():
                    encoder_hidden_states_res, ori_hidden_states_res, attn_mask_res, _, _ = encoder(x=x[res], d=k_batch) 
            else: 
                encoder_hidden_states_res, ori_hidden_states_res, attn_mask_res, quan_loss_res, log_dict_res = encoder(x=x[res], d=k_batch)
            enc_h_list.append(encoder_hidden_states_res)
            attn_mask_list.append(attn_mask_res)
            quan_loss_list.append(quan_loss_res)
            log_dict_list.append(log_dict_res)
                
        
        encoder_hidden_states = torch.stack(enc_h_list, dim=2).flatten(1,2)
        attn_mask = torch.stack(attn_mask_list, dim=2).flatten(1,2)
        #####
        log_dict = log_dict_list[-1]
        quan_loss = quan_loss_list[-1]

        noise = torch.randn_like(x)
        model_kwargs = dict(
            encoder_hidden_states=encoder_hidden_states,
            mask=attn_mask,
        )
        force_recon_loss = True 

     
        dm_model = self.model
        if self.diffusion_type == 'flow':
            loss_dict = self.diffusion.training_losses(
                dm_model, x, t, model_kwargs, noise=noise, recon_ratio=self.recon_ratio
            )
            batch_mse = loss_dict["loss"].mean()
        else:
            loss_dict = self.diffusion.training_losses(
                dm_model, x, t, model_kwargs, noise=noise,
                force_recon_loss=force_recon_loss, weighting=False
            )
            per_sample_mse_loss = mean_flat((x - loss_dict['pred_xstart']) ** 2)
            batch_mse = per_sample_mse_loss.mean()

        dm_loss = loss_dict["loss"].mean()

        loss = dm_loss + quan_loss
        log_dict["loss"] = loss.item()
        log_dict["mse"] = batch_mse.item()

        return loss, log_dict
        



