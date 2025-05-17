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
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from mimogpt.models.selftok.models import DiT, DiTBlock, get_2d_sincos_pos_embed, modulate, TimestepEmbedder, FinalLayer
import torch.nn.functional as F
from torch.nn import init
from mimogpt.models.selftok.sd3.sd3_utils import is_torch_npu_available
if is_torch_npu_available():
    import torch_npu


def modulate(x, shift, scale, dim=1):
    if shift is None or scale is None:
        return x
    return x * (1 + scale.unsqueeze(dim)) + shift.unsqueeze(dim)

def gate(x, gate):
    if gate is None:
        return x
    return gate.unsqueeze(0) * x

if is_torch_npu_available():
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim: int, elementwise_affine: bool = True, eps: float = 1e-6, dtype=torch.float, device='cpu'):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            x_dtype = x.dtype
            return torch_npu.npu_rms_norm(x.float(), self.weight.float(), epsilon=1e-6)[0].to(dtype=x_dtype)
else:
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim: int, elementwise_affine: bool = True, eps: float = 1e-6, dtype=torch.float, device='cpu'):
            """
            Initialize the RMSNorm normalization layer.
            Args:
                dim (int): The dimension of the input tensor.
                eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
            Attributes:
                eps (float): A small value added to the denominator for numerical stability.
                weight (nn.Parameter): Learnable scaling parameter.
            """
            super().__init__()
            self.eps = eps
            self.learnable_scale = elementwise_affine
            if self.learnable_scale:
                self.weight = nn.Parameter(torch.empty(dim).to(dtype=dtype, device=device))
            else:
                self.register_parameter("weight", None)
            self.reset_parameters()

        def reset_parameters(self,):
            if self.learnable_scale:
                init.ones_(self.weight)

        def _norm(self, x):
            """
            Apply the RMSNorm normalization to the input tensor.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The normalized tensor.
            """
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x):
            """
            Forward pass through the RMSNorm layer.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after applying RMSNorm.
            """
            x = self._norm(x)
            if self.learnable_scale:
                return x * self.weight.to(device=x.device, dtype=x.dtype)
            else:
                return x


class ViTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn=Attention, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DualAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            query_dim: int,
            num_heads: int = 8,
            query_heads: int=8,
            bidirectional: bool=True,
            zero_init: bool = False,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if bidirectional:
            assert dim == query_dim
            assert num_heads == query_heads
        self.num_heads = num_heads
        self.query_heads = query_heads
        self.head_dim = dim // num_heads
        self.query_head_dim = query_dim // query_heads
        self.bidrectional = bidirectional

        if qk_norm:
            print("Encoder using qk norm...")
            
        # latent linear
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_query_kv = nn.Linear(dim, query_dim * 2, bias=qkv_bias) \
            if (not bidirectional) or zero_init else nn.Identity()
        # query linear
        self.query_linear = nn.Linear(query_dim, query_dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.query_qnorm = norm_layer(self.query_head_dim) if qk_norm else nn.Identity()
        self.query_knorm = norm_layer(self.query_head_dim) if qk_norm else nn.Identity()

        self.zero_init = zero_init
        if self.zero_init:
            self.gate = torch.nn.Parameter(torch.zeros(1, self.query_heads, 1,1))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, query: torch.Tensor, mask: torch.Tensor=None, x_mask: torch.Tensor=None) -> torch.Tensor:
        B, N, C = x.shape
        _, query_N, query_C = query.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        query_qkv = self.query_linear(query).reshape(
            B, query_N, 3, self.query_heads, self.query_head_dim
        ).permute(2, 0, 3, 1, 4)

        if self.zero_init:
            kv = self.to_query_kv(x).reshape(B, N, 2, self.query_heads, self.query_head_dim).permute(2, 0, 3, 1, 4)
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=x_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            k, v = kv.unbind(0)
            query_q, query_k, query_v = query_qkv.unbind(0)
            xk = torch.cat([k, query_k], dim=2)
            xv = torch.cat([v, query_v], dim=2)
            query_q, xk = self.query_qnorm(query_q), self.query_knorm(xk)
            scale_factor = 1 / math.sqrt(self.query_head_dim)
            scores = query_q @ xk.transpose(2,3) * scale_factor
            if mask is not None:
                attn_bias = torch.zeros([B, self.query_heads, query_q.shape[2], xk.shape[2]], dtype=query_q.dtype, device=query_q.device)
                if mask.dtype == torch.bool:
                    attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
                else:
                    attn_bias += mask
                scores = scores + attn_bias
            scores = torch.cat(
                [
                    self.gate.tanh() * F.softmax(scores[:, :, :, :N], dim=-1).type_as(query_q),
                    F.softmax(scores[:, :, :, N:], dim=-1).type_as(query_q),
                ],
                dim=-1,
            ).type_as(query_q)
            scores = torch.dropout(scores, self.attn_drop.p if self.training else 0, train=True)
            query = scores @ xv
        elif self.bidrectional:
            query_q, query_k, query_v = query_qkv.unbind(0)
            query_q, query_k = self.query_qnorm(query_q), self.query_knorm(query_k)
            q = torch.cat((q, query_q), dim=2)
            k = torch.cat((k, query_k), dim=2)
            v = torch.cat((v, query_v), dim=2)
            x_cat = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x, query = x_cat[:,:, :N, :], x_cat[:,:, N:, :]
        else:
            kv = self.to_query_kv(x).reshape(B, N, 2, self.query_heads, self.query_head_dim).permute(2, 0, 3, 1, 4)
            if is_torch_npu_available():
                x = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    self.num_heads,
                    input_layout="BNSD",
                    pse=None,
                    atten_mask=x_mask if x_mask is None else torch.logical_not(x_mask), # attn_mask if attn_mask is None else torch.logical_not(attn_mask)
                    scale=1.0 / math.sqrt(q.shape[-1]),
                    pre_tockens=65536,
                    next_tockens=65536,
                    keep_prob=1.0,
                    sync=False,
                    inner_precise=0,
                )[0]
            else:
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=x_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            k, v = kv.unbind(0)
            query_q, query_k, query_v = query_qkv.unbind(0)

            k = torch.cat([k,query_k], dim=2)
            v = torch.cat([v,query_v], dim=2)
            q, k = self.query_qnorm(query_q), self.query_knorm(k)

            if is_torch_npu_available():
                query = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    self.query_heads,
                    input_layout="BNSD",
                    pse=None,
                    atten_mask=mask if mask is None else torch.logical_not(mask), # attn_mask if attn_mask is None else torch.logical_not(attn_mask)
                    scale=1.0 / math.sqrt(q.shape[-1]),
                    pre_tockens=65536,
                    next_tockens=65536,
                    keep_prob=1.0,
                    sync=False,
                    inner_precise=0,
                )[0]
            else:
                query = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )

        x = x.transpose(1, 2).reshape(B, N, C)
        query = query.transpose(1, 2).reshape(B, query_N, query_C)
        x = self.proj(x)
        query = self.query_proj(query)
        x = self.proj_drop(x)
        query = self.proj_drop(query)
        return x, query


class DualBlock(ViTBlock):
    """
    A dual block similar to SD3 setup.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, time_adaln=False, query_dim=256, diti=None, **block_kwargs):
        block_kwargs["query_dim"] = query_dim
        self.post_ln = block_kwargs.pop("post_ln", False)
        super().__init__(hidden_size, num_heads, mlp_ratio, attn=DualAttention, **block_kwargs)
        q_dim = query_dim
        self.q_norm1 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6) if not self.post_ln else nn.Identity()
        self.q_norm2 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6) if not self.post_ln else nn.Identity()
        self.post_norm1 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6) if self.post_ln else nn.Identity()
        self.post_norm2 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6) if self.post_ln else nn.Identity()
        self.ln_scale = 1.97 if self.post_ln else 1.0
        mlp_hidden_dim = int(q_dim * mlp_ratio)
        q_approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.q_mlp = Mlp(in_features=q_dim, hidden_features=mlp_hidden_dim, act_layer=q_approx_gelu, drop=0)
        self.time_adaln = time_adaln
        self.diti = diti
        if time_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(q_dim, 6 * q_dim, bias=True)
            )
            self.t_embedder = TimestepEmbedder(q_dim)
            self.init_block()

    def init_block(self):
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, q, mask=None):
        if self.time_adaln:
            K = q.shape[1]
            if self.diti is not None:
                pos_embed = self.diti.get_position(torch.arange(K).to(x.device))  # *7+1000
            else:
                pos_embed = torch.arange(K).to(x.device)
            t_emb = self.t_embedder(pos_embed)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None,None,None,None,None,None
        x_attn, q_attn = self.attn(self.norm1(x), modulate(self.q_norm1(q), shift_msa, scale_msa, 0), mask=mask)
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))

        q = self.post_norm1(self.ln_scale * q + gate(q_attn, gate_msa))
        q = self.post_norm2(self.ln_scale * q + gate(self.q_mlp(modulate(self.q_norm2(q), shift_mlp, scale_mlp, 0)), gate_mlp))
        return x, q


class DualAttentionMultiRes(nn.Module):
    def __init__(
            self,
            dim: int,
            query_dim: int,
            num_heads: int = 8,
            query_heads: int=8,
            bidirectional: bool=True,
            zero_init: bool = False,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = RMSNorm,
            low_res_dim: int = 64,  # multi-resolution specific params
            low_res_heads: int = 8, # multi-resolution specific params
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if bidirectional:
            assert dim == query_dim
            assert num_heads == query_heads
        self.num_heads = num_heads
        self.query_heads = query_heads
        self.low_res_heads = low_res_heads
        self.head_dim = dim // num_heads
        self.query_head_dim = query_dim // query_heads
        self.low_res_head_dim = low_res_dim // low_res_heads
        self.bidrectional = bidirectional

        if qk_norm:
            print("Encoder using qk norm...")
            
        # latent linear
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.x_to_query_kv = nn.Linear(dim, query_dim * 2, bias=qkv_bias) \
            if (not bidirectional) or zero_init else nn.Identity()
        # low resolution query linear
        self.low_res_linear = nn.Linear(low_res_dim, low_res_dim * 3, bias=qkv_bias)
        self.low_res_to_query_kv = nn.Linear(low_res_dim, query_dim * 2, bias=qkv_bias) \
            if (not bidirectional) or zero_init else nn.Identity()
        # query linear
        self.query_linear = nn.Linear(query_dim, query_dim * 3, bias=qkv_bias)

        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.query_qnorm = norm_layer(self.query_head_dim) if qk_norm else nn.Identity()
        self.query_knorm = norm_layer(self.query_head_dim) if qk_norm else nn.Identity()
        self.low_res_qnorm = norm_layer(self.low_res_head_dim) if qk_norm else nn.Identity()
        self.low_res_knorm = norm_layer(self.low_res_head_dim) if qk_norm else nn.Identity()

        self.zero_init = zero_init
        if self.zero_init:
            self.gate = torch.nn.Parameter(torch.zeros(1, self.query_heads, 1,1))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.low_res_proj = nn.Linear(low_res_dim, low_res_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, query: torch.Tensor, low_res: torch.Tensor=None, mask: torch.Tensor=None, x_mask: torch.Tensor=None, low_res_mask: torch.Tensor=None) -> torch.Tensor:
        # get dimension
        B, N, C = x.shape
        _, query_N, query_C = query.shape
        _, low_res_N, low_res_C = low_res.shape

        # image token
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # low_res hidden
        low_res_qkv = self.low_res_linear(low_res).reshape(B, low_res_N, 3, self.low_res_heads, self.low_res_head_dim).permute(2, 0, 3, 1, 4)
        q_low_res, k_low_res, v_low_res = low_res_qkv.unbind(0)
        q_low_res, k_low_res = self.low_res_qnorm(q_low_res), self.low_res_knorm(k_low_res)

        # query token
        query_qkv = self.query_linear(query).reshape(B, query_N, 3, self.query_heads, self.query_head_dim).permute(2, 0, 3, 1, 4)
        # no qk norm before KV append

        if self.zero_init:
            kv = self.to_query_kv(x).reshape(B, N, 2, self.query_heads, self.query_head_dim).permute(2, 0, 3, 1, 4)
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=x_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            k, v = kv.unbind(0)
            query_q, query_k, query_v = query_qkv.unbind(0)
            xk = torch.cat([k, query_k], dim=2)
            xv = torch.cat([v, query_v], dim=2)
            query_q, xk = self.query_qnorm(query_q), self.query_knorm(xk)
            scale_factor = 1 / math.sqrt(self.query_head_dim)
            scores = query_q @ xk.transpose(2,3) * scale_factor
            if mask is not None:
                attn_bias = torch.zeros([B, self.query_heads, query_q.shape[2], xk.shape[2]], dtype=query_q.dtype, device=query_q.device)
                if mask.dtype == torch.bool:
                    attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
                else:
                    attn_bias += mask
                scores = scores + attn_bias
            scores = torch.cat(
                [
                    self.gate.tanh() * F.softmax(scores[:, :, :, :N], dim=-1).type_as(query_q),
                    F.softmax(scores[:, :, :, N:], dim=-1).type_as(query_q),
                ],
                dim=-1,
            ).type_as(query_q)
            scores = torch.dropout(scores, self.attn_drop.p if self.training else 0, train=True)
            query = scores @ xv
        elif self.bidrectional:
            query_q, query_k, query_v = query_qkv.unbind(0)
            query_q, query_k = self.query_qnorm(query_q), self.query_knorm(query_k)
            q = torch.cat((q, query_q), dim=2)
            k = torch.cat((k, query_k), dim=2)
            v = torch.cat((v, query_v), dim=2)
            x_cat = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x, query = x_cat[:,:, :N, :], x_cat[:,:, N:, :]
        else:
            x_to_query_kv = self.x_to_query_kv(x).reshape(B, N, 2, self.query_heads, self.query_head_dim).permute(2, 0, 3, 1, 4)
            low_res_to_query_kv = self.low_res_to_query_kv(low_res).reshape(B, low_res_N, 2, self.query_heads, self.query_head_dim).permute(2, 0, 3, 1, 4)
            
            if is_torch_npu_available():
                x = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    self.num_heads,
                    input_layout="BNSD",
                    pse=None,
                    atten_mask=x_mask if x_mask is None else torch.logical_not(x_mask), # attn_mask if attn_mask is None else torch.logical_not(attn_mask)
                    scale=1.0 / math.sqrt(q.shape[-1]),
                    pre_tockens=65536,
                    next_tockens=65536,
                    keep_prob=1.0,
                    sync=False,
                    inner_precise=0,
                )[0]
            else:
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=x_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            

            if low_res_mask is None:
                torch.ones(low_res_N, low_res_N).tril().bool().unsqueeze(0).expand(B, 1, low_res_N, low_res_N).cuda()
            
            if is_torch_npu_available():
                low_res = torch_npu.npu_fusion_attention(
                    q_low_res,
                    k_low_res,
                    v_low_res,
                    self.low_res_heads,
                    input_layout="BNSD",
                    pse=None,
                    atten_mask=low_res_mask if low_res_mask is None else torch.logical_not(low_res_mask), # attn_mask if attn_mask is None else torch.logical_not(attn_mask)
                    scale=1.0 / math.sqrt(q_low_res.shape[-1]),
                    pre_tockens=65536,
                    next_tockens=65536,
                    keep_prob=1.0,
                    sync=False,
                    inner_precise=0,
                )[0]
            else:
                low_res = F.scaled_dot_product_attention(
                    q_low_res, k_low_res, v_low_res, attn_mask=low_res_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            

            x_to_query_k, x_to_query_v = x_to_query_kv.unbind(0)
            low_res_to_query_k, low_res_to_query_v = low_res_to_query_kv.unbind(0)
            query_q, query_k, query_v = query_qkv.unbind(0)

            k = torch.cat([x_to_query_k, low_res_to_query_k, query_k], dim=2)
            v = torch.cat([x_to_query_v, low_res_to_query_v, query_v], dim=2)
            q, k = self.query_qnorm(query_q), self.query_knorm(k)

            if is_torch_npu_available():
                query = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    self.query_heads,
                    input_layout="BNSD",
                    pse=None,
                    atten_mask=mask if mask is None else torch.logical_not(mask), # attn_mask if attn_mask is None else torch.logical_not(attn_mask)
                    scale=1.0 / math.sqrt(q.shape[-1]),
                    pre_tockens=65536,
                    next_tockens=65536,
                    keep_prob=1.0,
                    sync=False,
                    inner_precise=0,
                )[0]
            else:
                query = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )

        x = x.transpose(1, 2).reshape(B, N, C)
        low_res = low_res.transpose(1, 2).reshape(B, low_res_N, low_res_C)
        query = query.transpose(1, 2).reshape(B, query_N, query_C)
        
        x = self.proj(x)
        low_res = self.low_res_proj(low_res)
        query = self.query_proj(query)
        
        x = self.proj_drop(x)
        low_res = self.proj_drop(low_res)
        query = self.proj_drop(query)
        return x, query, low_res


class DualBlockMultiRes(ViTBlock):
    """
    A dual block similar to SD3 setup.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, time_adaln=False, query_dim=512, diti=None, low_res_hidden_dim=64, **block_kwargs):
        block_kwargs["query_dim"] = query_dim
        self.post_ln = block_kwargs.pop("post_ln", False)
        super().__init__(hidden_size, num_heads, mlp_ratio, attn=DualAttentionMultiRes, **block_kwargs)
        q_dim = query_dim
        self.q_norm1 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6) if not self.post_ln else nn.Identity()
        self.q_norm2 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6) if not self.post_ln else nn.Identity()
        self.post_norm1 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6) if self.post_ln else nn.Identity()
        self.post_norm2 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6) if self.post_ln else nn.Identity()

        self.low_res_norm1 = nn.LayerNorm(low_res_hidden_dim, elementwise_affine=False, eps=1e-6) if not self.post_ln else nn.Identity()
        self.low_res_norm2 = nn.LayerNorm(low_res_hidden_dim, elementwise_affine=False, eps=1e-6) if not self.post_ln else nn.Identity()
        self.low_res_post_norm1 = nn.LayerNorm(low_res_hidden_dim, elementwise_affine=False, eps=1e-6) if self.post_ln else nn.Identity()
        self.low_res_post_norm2 = nn.LayerNorm(low_res_hidden_dim, elementwise_affine=False, eps=1e-6) if self.post_ln else nn.Identity()

        self.ln_scale = 1.97 if self.post_ln else 1.0
        mlp_hidden_dim = int(q_dim * mlp_ratio)
        low_res_mlp_hidden_dim = int(low_res_hidden_dim * mlp_ratio)
        q_approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.q_mlp = Mlp(in_features=q_dim, hidden_features=mlp_hidden_dim, act_layer=q_approx_gelu, drop=0)
        self.low_res_mlp = Mlp(in_features=low_res_hidden_dim, hidden_features=low_res_mlp_hidden_dim, act_layer=q_approx_gelu, drop=0)
        self.time_adaln = time_adaln
        self.diti = diti
        if time_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(q_dim, 6 * q_dim, bias=True)
            )
            self.t_embedder = TimestepEmbedder(q_dim)

            self.adaLN_modulation_low_res = nn.Sequential(
                nn.SiLU(),
                nn.Linear(low_res_hidden_dim, 6 * low_res_hidden_dim, bias=True)
            )
            self.t_embedder_low_res = TimestepEmbedder(low_res_hidden_dim)

            self.init_block()

            self.init_block_low_res()

    def init_block(self):
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def init_block_low_res(self):
        nn.init.normal_(self.t_embedder_low_res.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_low_res.mlp[2].weight, std=0.02)
        nn.init.constant_(self.adaLN_modulation_low_res[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_low_res[-1].bias, 0)

    def forward(self, x, q, low_res=None, mask=None):
        if self.time_adaln:
            K = q.shape[1]
            K_low_res = low_res.shape[1] if low_res is not None else None
            
            if low_res != None:
                # if multi-resolution, then do not use get_position() *7+1000
                pos_embed_low_res = torch.arange(K_low_res).to(x.device)
                pos_embed = (torch.arange(K)+K_low_res).to(x.device)
            else:
                if self.diti is not None:
                    pos_embed = self.diti.get_position(torch.arange(K).to(x.device))  # *7+1000
                else:
                    pos_embed = torch.arange(K).to(x.device)

            t_emb = self.t_embedder(pos_embed)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)

            if low_res != None:
                t_emb_low_res = self.t_embedder_low_res(pos_embed_low_res)
                shift_msa_low_res, scale_msa_low_res, gate_msa_low_res, shift_mlp_low_res, scale_mlp_low_res, gate_mlp_low_res = self.adaLN_modulation_low_res(t_emb_low_res).chunk(6, dim=1)

        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None,None,None,None,None,None
            if low_res != None:
                shift_msa_low_res, scale_msa_low_res, gate_msa_low_res, shift_mlp_low_res, scale_mlp_low_res, gate_mlp_low_res = None,None,None,None,None,None

        if low_res != None:
            x_attn, q_attn, low_res_attn = self.attn(self.norm1(x), modulate(self.q_norm1(q), shift_msa, scale_msa, 0), modulate(self.low_res_norm1(low_res), shift_msa_low_res, scale_msa_low_res, 0), mask=mask)
        else:
            x_attn, q_attn = self.attn(self.norm1(x), modulate(self.q_norm1(q), shift_msa, scale_msa, 0), mask=mask)

        x = x + x_attn
        x = x + self.mlp(self.norm2(x))

        q = self.post_norm1(self.ln_scale * q + gate(q_attn, gate_msa))
        q = self.post_norm2(self.ln_scale * q + gate(self.q_mlp(modulate(self.q_norm2(q), shift_mlp, scale_mlp, 0)), gate_mlp))

        if low_res != None:
            low_res = self.low_res_post_norm1(self.ln_scale * low_res + gate(low_res_attn, gate_msa_low_res))
            low_res = self.low_res_post_norm2(self.ln_scale * low_res + gate(self.low_res_mlp(modulate(self.low_res_norm2(low_res), shift_mlp_low_res, scale_mlp_low_res, 0)), gate_mlp_low_res))

        return x, q, low_res


class ConcatBlock(ViTBlock):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, time_adaln=False, query_dim=256, diti=None,**block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio, attn=Attention)
        q_dim = query_dim
        self.q_norm1 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6)
        self.q_norm2 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6)
        self.time_adaln = time_adaln
        self.diti = diti
        if time_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(q_dim, 6 * q_dim, bias=True)
            )
            self.t_embedder = TimestepEmbedder(q_dim)
            self.init_block()

    def init_block(self):
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, q):
        if self.time_adaln:
            K = q.shape[1]
            if self.diti is not None:
                pos_embed = self.diti.get_position(torch.arange(K).to(x.device))
            else:
                pos_embed = torch.arange(K).to(x.device)
            t_emb = self.t_embedder(pos_embed)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None,None,None,None,None,None
        norm_q = modulate(self.q_norm1(q), shift_msa, scale_msa, 0)
        x_attn, q_attn = self.attn(torch.cat((self.norm1(x), norm_q), dim=1)).split([x.shape[1], q.shape[1]], dim=1)
        x = x + x_attn
        q = q + gate(q_attn, gate_msa)
        x = x + self.mlp(self.norm2(x))
        q = q + gate(self.mlp(modulate(self.q_norm2(q), shift_mlp, scale_mlp, 0)), gate_mlp)
        return x, q


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            c_dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(c_dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        B, N, C = x.shape
        kv = self.kv(c).reshape(B, c.shape[1], 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if mask is not None:
            attn_mask = torch.zeros([B, self.num_heads, q.shape[2], k.shape[2]], dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, q.shape[2], 1) == 0, float('-inf'))
        else:
            attn_mask = None
        
        if self.fused_attn:
            if is_torch_npu_available():
                x = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    self.num_heads,
                    input_layout="BNSD",
                    pse=None,
                    atten_mask=attn_mask if attn_mask is None else torch.logical_not(attn_mask),
                    scale=1.0 / math.sqrt(q.shape[-1]),
                    pre_tockens=65536,
                    next_tockens=65536,
                    keep_prob=1.0,
                    sync=False,
                    inner_precise=0,
                )[0]
            else:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn += attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class QFormer(nn.Module):
    """
    A compact QFormer implementation for processing the visual features from ViTBlock by Dongze Lian, 12 July 2024. 
    """
    def __init__(self, num_query_token, hidden_size, query_dim, num_heads, depth, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.num_query_token = num_query_token
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.qformer_blocks = nn.ModuleList([
            CrossAttention(query_dim, hidden_size, num_heads, qkv_bias=True, **block_kwargs) for _ in range(depth)
        ])
        mlp_hidden_dim = int(query_dim * mlp_ratio)
        q_approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.vision_proj = Mlp(
            in_features=query_dim, hidden_features=mlp_hidden_dim, act_layer=q_approx_gelu, drop=0
        )

    def forward(self, image_feats_to_qformer, query_tokens):
        for i, block in enumerate(self.qformer_blocks):
            query_tokens = block(query_tokens, image_feats_to_qformer)
        image_feats_to_vq = F.normalize(
            self.vision_proj(query_tokens), dim=-1
        )
        return image_feats_to_vq


class DiTCrossAttnBlock(DiTBlock):
    def __init__(self, hidden_size, encoder_hidden_size, num_heads, mlp_ratio=4.0, cross_modulate=True, **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio, **block_kwargs)
        self.cross_attn = CrossAttention(hidden_size, encoder_hidden_size, num_heads, qkv_bias=True, **block_kwargs)
        if cross_modulate:
            self.cross_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 3 * hidden_size, bias=True)
            )
        else:
            self.cross_modulation = None
        self.cross_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
    def forward(self, x, c, encoder_hidden_states, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.cross_modulation is not None:
            gate_mca, shift_mca, scale_mca = self.cross_modulation(c).chunk(3, dim=1)
            x = x + gate_mca.unsqueeze(1) * self.cross_attn(modulate(self.cross_norm(x), shift_mca, scale_mca), encoder_hidden_states, mask)
        else:
            x = x + self.cross_attn(self.cross_norm(x), encoder_hidden_states)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTDualBlock(DiTBlock):
    """
    A dual block similar to SD3 setup.
    """
    def __init__(self, hidden_size, q_dim, num_heads, query_heads, mlp_ratio=4.0, dit_attention='bi', **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio, **block_kwargs)
        self.q_norm1 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6)
        self.q_norm2 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6)
        approx_gelu_q = lambda: nn.GELU(approximate="tanh")
        mlp_hidden_dim = int(q_dim * mlp_ratio)
        self.q_mlp = Mlp(in_features=q_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu_q, drop=0)
        
        bidirectional = self.bidirectional = (dit_attention == 'bi')
        zero_init = (dit_attention == 'uni-0')
        self.adaLN_modulation_q = nn.Sequential(
            nn.SiLU(),
            nn.Linear(q_dim, 6 * q_dim, bias=True)
        )
        self.attn = DualAttention(
            dim=q_dim, query_dim=hidden_size, num_heads=num_heads, query_heads=query_heads, qkv_bias=True,
            bidirectional=bidirectional, zero_init=zero_init
        )

    def forward(self, x, t_emb, t_embd_q, q, mask=None): 
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        if self.bidirectional:
            shift_msa_q, scale_msa_q, gate_msa_q, shift_mlp_q, scale_mlp_q, gate_mlp_q = \
                self.adaLN_modulation_q(t_emb).chunk(6, dim=1)
            if mask is not None:
                x_mask = torch.ones((x.shape[0], x.shape[1])).to(x.device)
                mask = torch.cat((mask, x_mask), dim=1).bool().unsqueeze(1).unsqueeze(2)
                mask1 = None
        else:
            shift_msa_q, scale_msa_q, gate_msa_q, shift_mlp_q, scale_mlp_q, gate_mlp_q = \
                self.adaLN_modulation_q(t_embd_q).chunk(6, dim=1)
            if mask is not None:
                mask1 = mask.bool().unsqueeze(1).unsqueeze(2)
                x_mask = torch.ones((x.shape[0], x.shape[1])).to(x.device)
                mask = torch.cat((mask, x_mask), dim=1).bool().unsqueeze(1).unsqueeze(2)
            
        x_attn = modulate(self.norm1(x), shift_msa_x, scale_msa_x, 1)
        q_attn = modulate(self.q_norm1(q), shift_msa_q, scale_msa_q, 1)
        q_attn, x_attn = self.attn(x=q_attn, query=x_attn, mask=mask, x_mask=mask1)

        x_attn = x + gate_msa_x.unsqueeze(1) * x_attn
        q_attn = q + gate_msa_q.unsqueeze(1) * q_attn
        x = x + gate_mlp_x.unsqueeze(1) * self.mlp(modulate(self.norm2(x_attn), shift_mlp_x, scale_mlp_x, 1))
        q = q + gate_mlp_q.unsqueeze(1) * self.q_mlp(modulate(self.q_norm2(q_attn), shift_mlp_q, scale_mlp_q, 1))
        return x, q