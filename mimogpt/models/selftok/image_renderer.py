
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



from collections import OrderedDict
from .model_zoo import Enc_models, DiT_models
import torch
from mimogpt.utils import hf_logger
from torch import nn
from mimogpt.models.selftok.diti_utils import DiTi_cont, DiTi_normal
from mimogpt.models.selftok.sd3.rectified_flow import RectifiedFlow
import torch.nn.functional as F


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

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


class ImageRenderer(nn.Module):
    def __init__(
        self,
        image_size,
        k,
        encoder_hidden_size,
        enc,
        model,
        encoder_config,
        decoder_config,
        quantizer_config,
        k_m = None,
        k_s = None,
        stages =None,
        k_per_stage = None,
        noise_schedule_config=None,
        gradient_checkpointing=False,
        in_channels=16,
        enable_enc_variable_size=False,    # to enable variable image size for encoder; max size=MAX_LATENT_SIZE after downsampling by vae
        **kwargs,
    ):
        super().__init__()

        # reformat configs
        train_filter = decoder_config['train_filter']
        freeze_filter = decoder_config['freeze_filter']
        decoder_config['train_filter'] = train_filter.split('+') if train_filter != 'all' else None
        decoder_config['freeze_filter'] = freeze_filter.split('+') if freeze_filter != '' else []
        use_smart_react = quantizer_config.pop('smart_react')

        self.k_m = k_m
        self.k_s = k_s
        self.k = k

        # create model
        self.diffusion = RectifiedFlow(**noise_schedule_config)
        self.recon_ratio = 1.0      # reconstruction loss ratio (against velocity loss)
        assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        if stages is not None:
            self.diti = DiTi_cont(1000, k, stages, k_per_stage)
        else:
            self.diti = DiTi_normal(1000, self.k, self.k_m, self.k_s)
        latent_size = image_size // 8
        
        # modify configs
        if 'Qformer' in enc:
            if enable_enc_variable_size:
                encoder_config['pos_embed_max_size'] = 2 * latent_size
                encoder_config['diti'] = self.diti
                decoder_config['diti'] = self.diti
        else:
            if enable_enc_variable_size:
                assert False, "Other encoder does not support variable input size."
        
        if use_smart_react:
            quantizer_config["smart_re_K"] = self.k

        self.encoder = Enc_models[enc](
            K=self.k,
            input_size=latent_size,
            encoder_hidden_size=encoder_hidden_size,
            in_channels=in_channels,
            gradient_checkpointing=gradient_checkpointing,
            quantizer_config=quantizer_config,
            **encoder_config
        )

        self.model = DiT_models[model](
            K=self.k,
            input_size=latent_size, 
            encoder_hidden_size=encoder_hidden_size,
            in_channels=in_channels,
            gradient_checkpointing=gradient_checkpointing,
            **decoder_config
            )
    
        self.model.freeze()  # keep only params matching train_filter
        self.T = self.diffusion.num_timesteps

    def set_train(self):
        self.model.train()
        self.encoder.train()

    def set_eval(self):
        self.model.eval()
        self.encoder.eval()

    def forward(self, x, recon=True, **kwargs):
        k_batch = torch.ones(x.shape[0]).cuda() * (self.k - 1)

        # encode to get tokens
        if not self.encoder.training:
            with torch.no_grad():
                encoder_hidden_states, _, ori_hidden_states, attn_mask, _, _, ids = self.encoder(x=x, d=k_batch, kwargs=kwargs)
        else:
            encoder_hidden_states, _, ori_hidden_states, attn_mask, _, _, ids = self.encoder(x=x, d=k_batch, kwargs=kwargs)
       
        model_kwargs = dict(
            encoder_hidden_states=encoder_hidden_states,
            mask=attn_mask,
        )
        if recon:
            pred_x0 = self.model(x, **model_kwargs)
        else:
            pred_x0 = None

        return ids, ori_hidden_states, pred_x0
        