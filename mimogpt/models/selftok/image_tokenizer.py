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
import random
from .model_zoo import Enc_models, DiT_models
import torch
from mimogpt.utils import hf_logger
from torch import nn
from copy import deepcopy
from mimogpt.models.selftok.diti_utils import  DiTi_cont, DiTi_normal
from mimogpt.models.selftok.sd3.rectified_flow import RectifiedFlow
import torch.nn.functional as F

MAX_LATENT_SIZE = 384


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


class ImageTokenizer(nn.Module):
    def __init__(
        self,
        image_size,
        k,
        encoder_hidden_size,
        enc,
        model,
        ema_enc,
        enc_decay,
        L2_lr,
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
        diffusion_type='flow',
        enable_enc_variable_size=False,    # to enable variable image size for encoder; max size=MAX_LATENT_SIZE after downsampling by vae
        init_ratio=0,
        t2k = 1.,
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
        self.t2k = t2k

        self.alpha_multires_align_t = decoder_config.get("alpha_multires_align_t", None)

        # create model
        self.diffusion_type = diffusion_type
        assert diffusion_type == 'flow'
        self.diffusion = RectifiedFlow(**noise_schedule_config)
        self.recon_ratio = 1.0      # reconstruction loss ratio (against velocity loss)
        assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        if stages is not None:
            self.diti = DiTi_cont(1000, k, stages, k_per_stage)
        else:
            self.diti = DiTi_normal(1000, self.k, self.k_m, self.k_s)
        self.init_ratio = init_ratio
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

        enc_k = self.k 
        
        if use_smart_react:
            quantizer_config["smart_re_K"] = enc_k

        self.encoder = Enc_models[enc](
            K=enc_k,
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

        self.ema_enc = ema_enc
        if self.ema_enc:
            self.enc_ema = self.set_ema_model(self.encoder)
            self.enc_decay = enc_decay
            self.L2_lr = L2_lr

        self.context_see_xt = kwargs.get('context_see_xt', False)
        self.context_see_rec = kwargs.get('context_see_rec', False)
    
    def get_cdf(self, t):
        proj = torch.distributions.normal.Normal(self.k_m, self.k_s)
        t = torch.log(t / (1 - t))
        t = proj.cdf(t)
        return t

    def set_ema_model(self, model):
        ema = deepcopy(model).to(torch.float32)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        update_ema(ema, model, decay=0)
        ema = ema.cuda()
        ema.eval()
        return ema

    def set_train(self):
        self.model.train()
        self.encoder.train()

    def set_eval(self):
        self.model.eval()
        self.encoder.eval()

    def multires_align_t(self, t, alpha):
        t_lowres = (alpha * t) / (1+(alpha-1)*t)
        return t_lowres

    def batch_segment_uniform_sample(self, batch_size, stages, k_per_stages):
        samples = []
        for _ in range(batch_size):
            rand_value = random.random()
   
            K = sum(k_per_stages)
            k_acc = 0
            for i, k in enumerate(k_per_stages):
                k_acc += k
                if rand_value <= k_acc / K:
                    samples.append(random.uniform(stages[i]/1000, stages[i+1]/1000))
                    break

        samples = torch.tensor(samples)
        return samples
    
    def forward(self, x, full_tokens=False, **kwargs):
        # kwargs = {"token_ids":xx, "token_embeds":xx, "recons":xx}  # recon_shape 32*32*16
        low_res_latent = kwargs.get('low_res_latent', None)
        token_embeds = kwargs.get('token_embeds', None)
        # set correct shift
        if (x.shape[2] * x.shape[3] / 4096.0) < 0.5:
            shift = 1.0  # e^0.5
            high_res = False
        else:
            shift = 1.878     # e^0.63
            high_res = True
    

        t = torch.rand(x.shape[0]).cuda()

        # set number of tokens
        if self.k_m is None:
            if full_tokens:
                k_batch = self.diti.to_indices(torch.ones_like(t) * 1000.0)
            else:
                t_tmp = (self.t2k * t).clamp(0, 1.0)
                k_batch = self.diti.to_indices(t_tmp * 1000.0)
        else:
            if full_tokens:
                k_batch = self.diti.to_indices(torch.ones_like(t))
            else:
                t_tmp = (self.t2k * t).clamp(0, 1.0)
                k_batch = self.diti.to_indices(t_tmp)

        #new code for align timesteps on multires#
        if self.alpha_multires_align_t is not None:
            t_lowres = self.multires_align_t(t, self.alpha_multires_align_t)
            t_lowres_tmp = (self.t2k * t_lowres).clamp(0, 1.0)
            K_lowres = kwargs['token_embeds'].shape[1]
            k_lowres_batch = (self.diti.to_indices(t_lowres_tmp) * K_lowres / self.k).to(torch.long)
            context_lowres_mask = torch.arange(K_lowres).expand(k_lowres_batch.shape[0],K_lowres).to(k_lowres_batch.device)
            context_lowres_mask = context_lowres_mask <= k_lowres_batch.unsqueeze(1)
        else:
            context_lowres_mask = None
            
        #new code for align timesteps on multires#
        
        # shift t according to timestep
        t = self.diffusion.shift_t(t, shift)

        # encode to get tokens
        if not self.encoder.training:
            with torch.no_grad():
                encoder_hidden_states, to_quantizer_features, _, attn_mask, quan_loss, log_dict, _ = self.encoder(x=x, d=k_batch, kwargs=kwargs)
            to_quantizer_features_ema = None
        else:
            encoder_hidden_states, to_quantizer_features, _, attn_mask, quan_loss, log_dict, _ = self.encoder(x=x, d=k_batch, kwargs=kwargs)
            to_quantizer_features_ema = None

        encoder_hidden_states_d = encoder_hidden_states
        attn_mask_d = attn_mask

        # flow training
        noise = torch.randn_like(x)
        model_kwargs = dict(
            encoder_hidden_states=encoder_hidden_states_d,
            mask=attn_mask_d,
            low_res_latent=low_res_latent,
            hidden_states_low_res=token_embeds,
            context_see_xt = self.context_see_xt,
            context_see_rec = self.context_see_rec,
            context_lowres_mask = context_lowres_mask,
        )
        loss_dict = self.diffusion.training_losses(
            self.model, x, t, model_kwargs, noise=noise)
        # prepare logs
        batch_mse = loss_dict["loss"].sum() / loss_dict["loss"].shape[0]
        loss = batch_mse + quan_loss
        log_dict["loss"] = loss.item()
        log_dict["dm_mse"] = batch_mse.item()
        log_dict["loss_small"] = loss_dict["small"].item()
        log_dict["loss_mid"] = loss_dict["mid"].item()
        log_dict["loss_large"] = loss_dict["large"].item()
        log_dict["loss_uncon"] = loss_dict["uncon"].item()
        
        if self.ema_enc and to_quantizer_features_ema is not None:
            with torch.no_grad():
                L2_mask = attn_mask.clone()
                last_true_indices = torch.sum(L2_mask, dim=1)
                last_true_indices = last_true_indices - 1
                L2_mask[:, last_true_indices] = False
                L2_mask = L2_mask.unsqueeze(-1).expand(-1, -1, to_quantizer_features.shape[2])
            mt_1 = to_quantizer_features * L2_mask.float()
            mt_2 =to_quantizer_features_ema * L2_mask.float()
            L2_loss = F.mse_loss(mt_1, mt_2)
            loss += self.L2_lr * L2_loss
            log_dict["L2_loss"] = self.L2_lr * L2_loss.item()
        
        return loss, log_dict
        