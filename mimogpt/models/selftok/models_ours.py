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

import torch
import torch.nn as nn
import numpy as np
import math
from .sd3.mmdit import PatchEmbed, get_1d_sincos_pos_embed_from_grid
from mimogpt.models.selftok.models import DiT, DiTBlock, get_2d_sincos_pos_embed, modulate, TimestepEmbedder, FinalLayer
import torch.nn.functional as F
from .quantizer import construct_quantizer
from .modules import DiTCrossAttnBlock, ViTBlock, QFormer, DualBlock, ConcatBlock, DiTDualBlock, DualBlockMultiRes
from einops import rearrange
import torch.distributed as dist
import random

try:
    from torch.utils.checkpoint import checkpoint
    print("Using gradient checkpointing...")
except:
    print("Disabling gradient checkpointing...")
    assert False

from torch.utils.checkpoint import checkpoint
def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs
    return ckpt_forward

class Encoder(nn.Module):
    def __init__(
        self, K, input_size=32, encoder_hidden_size=256, patch_size=8, in_channels=4,
        hidden_size=256, depth=None, num_heads=4, mlp_ratio=4.0,
        pre_norm=False, post_norm=True, encoder_out_dim=None, apply_losses_together=False,
        gradient_checkpointing=False, pos_embed_max_size=None, quantizer_config=None, attn_mask=False, single_token=False, **kwargs
    ):
        super().__init__()
        self.K = K
        self.n_e = quantizer_config['codebook_size']
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        depth = depth or self.K
        self.depth = depth
        self.hidden_size = hidden_size
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.pos_embed_max_size = pos_embed_max_size
        encoder_out_dim = encoder_out_dim or hidden_size
        self.gradient_checkpointing = gradient_checkpointing
        self.code_dim = quantizer_config['code_dim']
        self.n_tokens = K * (input_size // patch_size) ** 2
        self.apply_losses_together = apply_losses_together
        self.attn_mask = attn_mask
        self.single_token = single_token

        # models
        self.x_embedder = PatchEmbed(img_size=input_size,patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size, bias=True)
        if pos_embed_max_size is not None:
            num_patches = pos_embed_max_size * pos_embed_max_size
            self.x_embedder.num_patches = pos_embed_max_size * pos_embed_max_size
        else:
            num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        if num_patches is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        else:
            self.pos_embed = None
        self.blocks = nn.ModuleList([
            ViTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer_norm = nn.LayerNorm(encoder_out_dim, eps=1e-6)
        self.final_layer_norm2 = nn.LayerNorm(self.code_dim, eps=1e-6)
        self.final_layer_norm3 = nn.LayerNorm(encoder_hidden_size, eps=1e-6) 

        self.quantizer = construct_quantizer(
            latent_dim = encoder_out_dim,
            output_dim = encoder_hidden_size,
            **quantizer_config
        )
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.x_embedder.num_patches ** 0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
            
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward_quantizer(self, quantizer, x):
        outs_q, indices, loss, log_dict = quantizer(x)
        return outs_q, indices, loss, log_dict
    
    def get_encoder_outs(self, x, kwargs=None):
        outs = []
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing:
                x = checkpoint(ckpt_wrapper(block), x, use_reentrant=False)
            else:
                x = block(x)
            if i >= self.depth - self.K:
                outs.append(x)
    
        assert len(outs) == self.K
        outs = torch.cat(outs, dim=1)
        return outs

    def get_encoder_mask(self, x, d):
        B, N, P = x.shape[0], self.K, x.shape[1]
        enc_mask = torch.arange(self.K).repeat_interleave(P)[None, ...].expand(B,N).to(d.device)
        return (enc_mask <= d.unsqueeze(1))
    
    def calc_entropy(self, p):
        ap = p.mean(dim=0)
        p_log_p = ap * torch.log(ap)
        entropy_to_max = -p_log_p.sum(dim=-1)
        # E(H(p))
        p_log_p = p * torch.log(p)
        entropy_to_min = -p_log_p.sum(dim=-1)
        entropy_to_min = entropy_to_min
        return entropy_to_min
    
    def get_perplexity_list(self, log_dict, chunks=50):
        if 'perplexity_list' in log_dict:
            # separate codebook
            perplexity_list = torch.tensor(log_dict['perplexity_list'])
            perplexity_list = torch.stack([t.mean(dim=0) for t in perplexity_list.tensor_split(chunks, dim=0)],dim=0).float()
            deter_list = torch.tensor(log_dict['deter_list']).float()
            deter_list = torch.stack([t.mean(dim=0) for t in deter_list.tensor_split(chunks, dim=0)],dim=0).float()
            return perplexity_list.tolist(), deter_list.tolist()

        probs = self.quantizer._codebook.timestep_p_over_c.mean(dim=0)
        chunk_probs = torch.stack([t.mean(dim=0) for t in probs.tensor_split(chunks, dim=0)],dim=0).float()
        ap = chunk_probs
        perplexity_list = torch.exp(-torch.sum(ap * torch.log(ap + 1e-10), dim=1)).tolist()
        deterministic_list = self.calc_entropy(ap).tolist()
        return perplexity_list, deterministic_list

    def cropped_pos_embed(self, hw):
        assert self.pos_embed_max_size is not None
        p = self.x_embedder.patch_size[0] #(2,2)
        h, w = hw
        # patched size
        h = h // p
        w = w // p
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(
            self.pos_embed,
            "1 (h w) c -> 1 h w c",
            h=self.pos_embed_max_size,
            w=self.pos_embed_max_size,
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, "1 h w c -> 1 (h w) c")
        return spatial_pos_embed
    
    def forward(self, x=None, hidden_states=None, d=None, kwargs=None):
        """
        Forward pass of feature encoder.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        d: N, the depth for each sample
        """
        
        if self.pos_embed_max_size is not None:
            hw = x.shape[-2:]
            x = self.x_embedder(x) #torch.Size([4, 256, 64])
            x = x + self.cropped_pos_embed(hw) #torch.Size([1, 256, 64])
        else:
            x = self.x_embedder(x) + self.pos_embed
        if hidden_states is None:
            outs = self.get_encoder_outs(x, kwargs=kwargs) #torch.Size([4, 512, 512])
            if self.pre_norm:
                outs = self.final_layer_norm(outs) 
            to_quantizer_features = outs
            perplexity_list = []
            deterministic_list = []

            if self.apply_losses_together:  # False
                enc_mask = self.get_encoder_mask(x, d)
                grad_mask = enc_mask[..., None].expand_as(to_quantizer_features).float()
                to_quantizer_features = to_quantizer_features * grad_mask + \
                    to_quantizer_features.detach() * (1-grad_mask)
            
            outs_q, indices, loss, log_dict = \
                self.forward_quantizer(self.quantizer, to_quantizer_features)
            
            # prepare logs
            perplexity_list, deterministic_list = self.get_perplexity_list(log_dict)
            log_dict.update({
                "perplexity_list": perplexity_list,
                "deter_list": deterministic_list,
            })

            if self.post_norm:
                outs_q = self.final_layer_norm3(outs_q)
        else:
            outs_q = hidden_states
            loss = 0.0
            log_dict = {}
            to_quantizer_features = None
            indices = None
        
        if d is None:
            return outs_q, indices
        
        enc_mask = self.get_encoder_mask(x, d)
        attn_mask = enc_mask
        mask_v = enc_mask[..., None].expand_as(outs_q)
        encoder_hidden_states = outs_q * mask_v
        return encoder_hidden_states, to_quantizer_features, outs_q, attn_mask, loss, log_dict, indices
    

'''
    Encoder with a special input: mode, which can be either
        - 'qformer' with cross attention interaction between query and latent
        - 'concat' with self attention interaction between query and latent
        - 'dual-xx' with self attention interaction between query and latent, but query has its own transformer
            - xx='cross': query as q, latent as kv
            - xx='self': [query,latent] into self-attention
'''
class QformerEncoder(Encoder):
    def __init__(
        self, K, input_size=32, encoder_hidden_size=256, patch_size=8, in_channels=4,
        hidden_size=256, depth=None, num_heads=4, mlp_ratio=4.0,
        pre_norm=False, post_norm=True, qformer_mode='qformer',
        gradient_checkpointing=False, pos_embed_max_size=None, apply_losses_together=False,
        xavier_init=False, diti=None, quantizer_config=None, attn_mask=False, single_token=False, **kwargs
    ):
        super().__init__(
            K, input_size, encoder_hidden_size, patch_size, in_channels, hidden_size, depth, num_heads,
            mlp_ratio, pre_norm, post_norm, encoder_out_dim=kwargs['query_dim'],
            gradient_checkpointing=gradient_checkpointing, apply_losses_together=apply_losses_together,
            pos_embed_max_size=pos_embed_max_size, quantizer_config=quantizer_config, attn_mask=attn_mask, single_token=single_token, **kwargs
        )
        qformer_depth = depth
        self.num_query_token = K # num_query_token
        query_dim = kwargs['query_dim']
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_query_token, query_dim))
        self.query_tokens.data.normal_(mean=0.0, std=0.02) #initialization
        self.mode = qformer_mode
        self.diti = diti
        self.attn_mask = attn_mask
        self.single_token = single_token
        if diti:
            kwargs["diti"] = diti
        if self.mode == 'qformer':
            self.qformer = QFormer(
                self.num_query_token, hidden_size, query_dim, num_heads, qformer_depth, mlp_ratio=mlp_ratio
            )
            self.blocks = nn.Identity()
        elif self.mode == 'dual':
            self.blocks = nn.ModuleList([
                DualBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)
            ])
        elif self.mode == 'concat':
            self.blocks = nn.ModuleList([
                ConcatBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)
            ])

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        if xavier_init:
            self.apply(_basic_init)

    def get_encoder_outs(self, x, kwargs=None):
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
        if self.mode == 'qformer':
            query_tokens = self.qformer(x, query_tokens) # [B, L, C]
        elif self.mode == 'concat':
            for i, block in enumerate(self.blocks):
                if self.gradient_checkpointing:
                    x, query_tokens = checkpoint(ckpt_wrapper(block), x, query_tokens, use_reentrant=False)
                else:
                    x, query_tokens = block(x, query_tokens)
        elif self.mode == 'dual':
            # attn mask
            
            if self.attn_mask: #False
                mask = mask = torch.ones(self.K, self.K).tril().bool().cuda()
                x_mask = torch.ones((self.K, x.shape[1])).cuda()
                mask = torch.cat((x_mask, mask), dim=1).bool()
                mask = mask.unsqueeze(0).unsqueeze(1).repeat(x.shape[0],1,1,1)
            else:
                mask = None

            for i, block in enumerate(self.blocks):
                if self.gradient_checkpointing:
                    x, query_tokens = checkpoint(ckpt_wrapper(block), x, query_tokens, mask, use_reentrant=False)
                else:
                    x, query_tokens = block(x, query_tokens, mask=mask)
        else:
            raise ValueError("Unknown mode to QFormerEncoder.")
        return query_tokens
    
    def get_encoder_mask(self, x, d, single_token=False):
        # no spatial token, so num patches is essentially 1
        B, N = x.shape[0], self.K
        enc_mask = torch.arange(self.K).repeat_interleave(1)[None, ...].expand(B,N).to(d.device)
        
        if single_token:
            return (enc_mask == d.unsqueeze(1))
        else:
            return (enc_mask <= d.unsqueeze(1))

class QformerEncoderMultiRes(QformerEncoder):
    def __init__(
        self, K, input_size=32, encoder_hidden_size=256, patch_size=8, in_channels=4,
        hidden_size=256, depth=None, num_heads=4, mlp_ratio=4.0,
        pre_norm=False, post_norm=True, qformer_mode='qformer',
        gradient_checkpointing=False, pos_embed_max_size=None, apply_losses_together=False,
        xavier_init=False, diti=None, quantizer_config=None, attn_mask=False, single_token=False,
        low_res_hidden_size=64, low_res_code_dim=16, low_res_codebook_size=32768, # multi-res specific parameters
        reuse_token_embeds=True, low_res_causal_mask=True, low_res_K=512, **kwargs   # multi-res specific parameters
    ):
        super().__init__(
            K, input_size, encoder_hidden_size, patch_size, in_channels, hidden_size, depth, num_heads,
            mlp_ratio, pre_norm, post_norm, qformer_mode=qformer_mode,
            gradient_checkpointing=gradient_checkpointing, apply_losses_together=apply_losses_together,
            pos_embed_max_size=pos_embed_max_size, quantizer_config=quantizer_config, attn_mask=attn_mask, single_token=single_token, **kwargs
        )
        assert self.mode == 'dual'
        self.blocks = nn.ModuleList([
                DualBlockMultiRes(hidden_size, num_heads, mlp_ratio=mlp_ratio, low_res_hidden_dim=low_res_hidden_size, **kwargs) for _ in range(depth)
            ])
        self.reuse_token_embeds = reuse_token_embeds
        self.low_res_causal_mask = low_res_causal_mask
        if self.reuse_token_embeds:
            if low_res_code_dim != low_res_hidden_size:
                self.token_embedder = nn.Linear(low_res_code_dim, low_res_hidden_size)
            else:
                self.token_embedder = nn.Identity()
        else:
            self.token_embedder = nn.Embedding(low_res_codebook_size, low_res_hidden_size)

        low_res_hidden_len = low_res_K
        low_res_hidden_pos_embed_hidden_size = low_res_hidden_size
        self.register_buffer(
            "low_res_hidden_pos_embed",
            torch.zeros(1, low_res_hidden_len, low_res_hidden_pos_embed_hidden_size)
        )
        pos_embed = np.arange(low_res_hidden_len, dtype=np.float32)
        
        low_res_hidden_pos_embed = get_1d_sincos_pos_embed_from_grid( # TODO
            low_res_hidden_pos_embed_hidden_size, pos_embed
        )
        
        self.low_res_hidden_pos_embed = torch.from_numpy(low_res_hidden_pos_embed).float().unsqueeze(0)

    def get_encoder_outs(self, x, kwargs=None):
        '''
            kwargs['token_ids']: B * N, where N is the num token of lower res
            kwargs['token_embeds']: B * N * D, where D is the code dim
        '''
        low_res_in = kwargs['token_embeds'] if self.reuse_token_embeds else kwargs['token_ids']  # B * N * D = 16 * 512 * 16

        low_res_hidden = self.token_embedder(low_res_in)  # B * N_low_res * hidden_size = 16 * 512 * 64
        low_res_hidden = low_res_hidden + self.low_res_hidden_pos_embed

        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)  # B * N * query_dim = 16 * 512 * 512
        if self.mode == 'qformer':
            query_tokens = self.qformer(x, query_tokens) # [B, L, C]
        elif self.mode == 'concat':
            for i, block in enumerate(self.blocks):
                if self.gradient_checkpointing:
                    x, query_tokens = checkpoint(ckpt_wrapper(block), x, query_tokens, use_reentrant=False)
                else:
                    x, query_tokens = block(x, query_tokens)
        elif self.mode == 'dual':
            if self.attn_mask:
                mask = mask = torch.ones(self.K, self.K).tril().bool().cuda()
                x_mask = torch.ones((self.K, x.shape[1])).cuda()
                low_res_mask = torch.ones((self.K, low_res_in.shape[1])).cuda()
                mask = torch.cat((x_mask, low_res_mask, mask), dim=1).bool()
                mask = mask.unsqueeze(0).unsqueeze(1).repeat(x.shape[0],1,1,1)
            else:
                mask = None
            to_print = random.uniform(0,1) < 0.05 and dist.get_rank() == 0
            for i, block in enumerate(self.blocks):
                x, query_tokens, low_res_hidden = checkpoint(ckpt_wrapper(block), x, query_tokens, low_res_hidden, mask, use_reentrant=False)
                if to_print and (i < 4 or i > 12):
                    print(f"Encoder Layer {i}: max(x)={x.abs().max(dim=-1)[0].mean().item()}, mean(x)={x.abs().mean(dim=-1).mean().item()};\
                                               max(q)={query_tokens.abs().max(dim=-1)[0].mean().item()}, mean(q)={query_tokens.abs().mean(dim=-1).mean().item()};\
                                               max(low_res)={low_res_hidden.abs().max(dim=-1)[0].mean().item()}, mean(low_res)={low_res_hidden.abs().mean(dim=-1).mean().item()}")
        else:
            raise ValueError("Unknown mode to QFormerEncoder.")
        return query_tokens
        
    def get_encoder_mask(self, x, d, single_token=False):
        B, N = x.shape[0], self.K
        enc_mask = torch.arange(self.K).repeat_interleave(1)[None, ...].expand(B,N).to(d.device)
        
        if single_token:
            return (enc_mask == d.unsqueeze(1))
        else:
            return (enc_mask <= d.unsqueeze(1))