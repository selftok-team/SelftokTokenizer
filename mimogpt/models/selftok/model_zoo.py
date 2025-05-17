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
from .models_ours import QformerEncoderMultiRes, QformerEncoder, Encoder
from .sd3.mmdit import MMDiT, MMDiT_Renderer
from .sd3.renderdit import RenderDiT


def MMDiT_XL(**kwargs):
    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {"in_features": kwargs['encoder_hidden_size'], "out_features": 1536},
    }
    diffusion_model = MMDiT(
        pos_embed_scaling_factor=None,
        pos_embed_offset=None,
        pos_embed_max_size=192,
        patch_size=2,
        depth=24,
        num_patches=36864,
        adm_in_channels=kwargs['encoder_hidden_size'],
        context_embedder_config=context_embedder_config,
        device='cpu',
        dtype=torch.float,
        **kwargs
    )
    return diffusion_model

def MMDiT_XL_Renderer(**kwargs):
    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {"in_features": kwargs['encoder_hidden_size'], "out_features": 1536},
    }
    diffusion_model = MMDiT_Renderer(
        pos_embed_scaling_factor=None,
        pos_embed_offset=None,
        pos_embed_max_size=192,
        patch_size=2,
        depth=24,
        num_patches=36864,
        adm_in_channels=kwargs['encoder_hidden_size'],
        context_embedder_config=context_embedder_config,
        device='cpu',
        dtype=torch.float,
        **kwargs
    )
    return diffusion_model

def RenderDiT_XL(**kwargs):
    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {"in_features": kwargs['encoder_hidden_size'], "out_features": 1536},
    }
    diffusion_model = RenderDiT(
        pos_embed_scaling_factor=None,
        pos_embed_offset=None,
        pos_embed_max_size=192,
        patch_size=2,
        depth=24,
        num_patches=36864,
        adm_in_channels=kwargs['encoder_hidden_size'],
        context_embedder_config=context_embedder_config,
        device='cpu',
        dtype=torch.float,
        **kwargs
    )
    return diffusion_model

def Enc_Tiny_8(**kwargs):
    return Encoder(patch_size=8, hidden_size=256, num_heads=4, **kwargs)

def Enc_Base_8(**kwargs):
    return Encoder(patch_size=8, hidden_size=768, num_heads=12, **kwargs)

def Enc_Base_16(**kwargs):
    return Encoder(patch_size=16, hidden_size=256, num_heads=4, **kwargs)

def Enc_L_8(**kwargs):
    assert kwargs["K"] <= 24, "Enc-L/8 supports K up to 24."
    return Encoder(patch_size=8, hidden_size=768, num_heads=16, depth=24, **kwargs)

def Enc_H_8(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/8 supports K up to 32."
    return Encoder(patch_size=8, hidden_size=768, num_heads=16, depth=32, **kwargs)

def Enc_H_8_XS(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/8 supports K up to 32."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=32, **kwargs)

def Enc_H_8_XS_24(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/8 supports K up to 32."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=24, **kwargs)

def Enc_H2_8_XS(**kwargs):
    assert kwargs["K"] <= 40, "Enc-H/8 supports K up to 40."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=40, **kwargs)

def Enc_H3_8_XS(**kwargs):
    assert kwargs["K"] <= 48, "Enc-H/8 supports K up to 48."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=48, **kwargs)

def Enc_B_8_XS(**kwargs):
    assert kwargs["K"] <= 16, "Enc-B/8 supports K up to 16."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=16, **kwargs)

def Enc_H_4_XS(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/4 supports K up to 32."
    return Encoder(patch_size=4, hidden_size=64, num_heads=8, depth=32, **kwargs)

def Enc_B_4_XS(**kwargs):
    assert kwargs["K"] <= 16, "Enc-B/4 supports K up to 16."
    return Encoder(patch_size=4, hidden_size=64, num_heads=8, depth=16, **kwargs)

def Enc_H_8_XXS(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/8 supports K up to 32."
    return Encoder(patch_size=8, hidden_size=128, num_heads=8, depth=32, **kwargs)

def Enc_Qformer_Bi_L_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=16, num_heads=2, depth=24,
        query_dim=16, query_heads=2, bidirectional=True, **kwargs
    )

def Enc_Qformer_Bi_WL_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=128, num_heads=4, depth=24,
        query_dim=128, query_heads=4, bidirectional=True, **kwargs
    )
    
def Enc_Qformer_Bi_UWL_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=256, num_heads=8, depth=24,
        query_dim=256, query_heads=8, bidirectional=True, **kwargs
    )

def Enc_Qformer_Bi_WL_1(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=128, num_heads=4, depth=24,
        query_dim=128, query_heads=4, bidirectional=True, **kwargs
    )

def Enc_Qformer_Bi_UWL_1(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=256, num_heads=8, depth=24,
        query_dim=256, query_heads=8, bidirectional=True, **kwargs
    )

def Enc_Qformer_Bi_XL_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=512, num_heads=4, depth=16,
        query_dim=512, query_heads=4, bidirectional=True, **kwargs
    )

def Enc_Qformer_Uni_WL_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=128, num_heads=4, depth=24,
        query_dim=256, query_heads=8, bidirectional=False, **kwargs
    )
    
def Enc_Qformer_Uni_M_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=64, num_heads=4, depth=16,
        query_dim=64, query_heads=4, bidirectional=False, **kwargs
    )


def Enc_Qformer_Uni_L_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=64, num_heads=4, depth=20,
        query_dim=128, query_heads=8, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_XL_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=64, num_heads=4, depth=16,
        query_dim=512, query_heads=8, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_WXL_3(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=256, num_heads=4, depth=28,
        query_dim=512, query_heads=4, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_WXL_4(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=256, num_heads=4, depth=28,
        query_dim=512, query_heads=4, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_WXL_5(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=256, num_heads=4, depth=28,
        query_dim=512, query_heads=8, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_XL_1(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=64, num_heads=4, depth=24,
        query_dim=256, query_heads=8, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_L2_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=128, num_heads=4, depth=24,
        query_dim=128, query_heads=4, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_WL_1(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=128, num_heads=4, depth=24,
        query_dim=256, query_heads=8, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_WXL_1(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=256, num_heads=4, depth=28,
        query_dim=256, query_heads=4, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_WXL_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=256, num_heads=4, depth=28,
        query_dim=256, query_heads=4, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni0_WL_1(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=128, num_heads=4, depth=24,
        query_dim=256, query_heads=8, bidirectional=False, zero_init=True, **kwargs
    )

def Enc_Qformer_Uni_UWL_1(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=256, num_heads=8, depth=24,
        query_dim=256, query_heads=8, bidirectional=False, **kwargs
    )

def Enc_Qformer_Multi_Res_Uni_XL_2(**kwargs):
    return QformerEncoderMultiRes(patch_size=2, hidden_size=64, num_heads=4, depth=16,
        query_dim=512, query_heads=8, bidirectional=False, low_res_hidden_size=64, 
        low_res_code_dim=16, low_res_codebook_size=32768, reuse_token_embeds=True, 
        low_res_causal_mask=True, low_res_K=512, **kwargs
    )

DiT_models = {
    'MMDiT_XL': MMDiT_XL,
    'RenderDiT_XL': RenderDiT_XL,
    "MMDiT_XL_Renderer": MMDiT_XL_Renderer,
}

Enc_models = {
    'Enc-Tiny/8': Enc_Tiny_8,
    'Enc-Base/8': Enc_Base_8,
    'Enc-L/8': Enc_L_8,
    'Enc-H/8': Enc_H_8,
    'Enc-H/8-XS': Enc_H_8_XS,
    'Enc-H/8-XS-24': Enc_H_8_XS_24,
    'Enc-H2/8-XS': Enc_H2_8_XS,
    'Enc-H3/8-XS': Enc_H3_8_XS,
    'Enc-B/8-XS': Enc_B_8_XS,
    'Enc-H/4-XS': Enc_H_4_XS,
    'Enc-B/4-XS': Enc_B_4_XS,
    'Enc-H/8-XXS': Enc_H_8_XXS,
    'Enc-Base/16': Enc_Base_16,
    'Enc-Qformer-Bi-L/2': Enc_Qformer_Bi_L_2,
    'Enc-Qformer-Bi-WL/2': Enc_Qformer_Bi_WL_2,
    'Enc-Qformer-Bi-UWL/2': Enc_Qformer_Bi_UWL_2,
    'Enc-Qformer-Bi-WL/1': Enc_Qformer_Bi_WL_1,
    'Enc-Qformer-Bi-UWL/1': Enc_Qformer_Bi_UWL_1,
    'Enc-Qformer-Uni-M/2': Enc_Qformer_Uni_M_2,
    'Enc-Qformer-Uni-L/2': Enc_Qformer_Uni_L_2,
    'Enc-Qformer-Uni-XL/2': Enc_Qformer_Uni_XL_2,
    'Enc-Qformer-Uni-XL/1': Enc_Qformer_Uni_XL_1,
    'Enc-Qformer-Uni-L2/2': Enc_Qformer_Uni_L2_2,
    'Enc-Qformer-Uni-WL/2': Enc_Qformer_Uni_WL_2,
    'Enc-Qformer-Uni-WL/1': Enc_Qformer_Uni_WL_1,
    'Enc-Qformer-Bi-XL/2': Enc_Qformer_Bi_XL_2,
    'Enc-Qformer-Uni-WXL/1': Enc_Qformer_Uni_WXL_1,
    'Enc-Qformer-Uni-WXL/2': Enc_Qformer_Uni_WXL_2,
    'Enc-Qformer-Uni-WXL/3': Enc_Qformer_Uni_WXL_3,
    'Enc-Qformer-Uni-WXL/4': Enc_Qformer_Uni_WXL_4,
    'Enc-Qformer-Uni-WXL/5': Enc_Qformer_Uni_WXL_5,
    'Enc-Qformer-Uni0-WL/1': Enc_Qformer_Uni0_WL_1,
    'Enc-Qformer-Uni-UWL/1': Enc_Qformer_Uni_UWL_1,
    'Enc-Qformer-Multi-Res-Uni-XL/2': Enc_Qformer_Multi_Res_Uni_XL_2,
}


