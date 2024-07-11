""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union
from collections import OrderedDict
import re
import os
import sys
from sklearn.metrics import pairwise
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch import Tensor
import open_clip.utils.misc as misc
import argparse
from functools import partial

from open_clip.utils.env import checkpoint_pathmgr as pathmgr
from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer, VisionTransformer_Mul
from .new_utils import to_2tuple
import time

from image_display import save_images_and_localisation

import time
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import torchvision.utils as vutils
from .vp import (
    PadPrompter,
    RandomPatchPrompter,
    FixedPatchPrompter
)

from torch.autograd import Variable, grad

PROMPT_TYPES = {
    "padding": PadPrompter,
    "random_patch": RandomPatchPrompter,
    "fixed_patch": FixedPatchPrompter
}
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__




@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = True

    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype



def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return visual

def _build_vision_tower_Mul(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer_Mul(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return visual

def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            output_tokens=text_cfg.output_tokens,
            pad_id=text_cfg.pad_id,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text

class BatchNormPoint(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.feat_size = feat_size
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        x = x.view(s1 * s2, self.feat_size)
        x = self.bn(x)
        return x.view(s1, s2, s3)

class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()

class TransformerBasicHead(nn.Module):
    """
    Basic Transformer Head. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes
    ):
        super(TransformerBasicHead, self).__init__()
        self.projection1 = nn.Linear(dim_in, 128, bias=True)
        self.projection2 = nn.Linear(128, 64, bias=True)
        self.projection3 = nn.Linear(64, num_classes, bias=True)
        self.bn1 = nn.BatchNorm1d(dim_in)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.projection1(x)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)
        x = self.projection2(x)
        x = F.relu(x, inplace=True)
        x = self.bn3(x)
        x = self.projection3(x)
        return torch.sigmoid(x)

# class ConvOperationLayer(nn.Module):
#     def __init__(self):
#         super(ConvOperationLayer, self).__init__()                                                          # ????????????????
#         self.conv1 = nn.Conv2d(in_channels=896, out_channels=512, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
#         self.fc = nn.Linear(128 * 225, 1)  # Assuming output is a single localization result

#     def forward(self, query_patches, normal_patches):
#         # Process query image patches
#         query_patches = query_patches.view(-1, 896, 225, 32)  # Reshape to match Conv2d input
#         q = self.conv1(query_patches)
#         q = torch.relu(q)
#         q = self.conv2(q)
#         q = torch.relu(q)
#         q = self.conv3(q)
#         q = torch.relu(q)

#         # Process normal image patches
#         normal_patches = normal_patches.view(-1, 896, 225, 8 * 32)  # Combine batch and normal patches
#         n = self.conv1(normal_patches)
#         n = torch.relu(n)
#         n = self.conv2(n)
#         n = torch.relu(n)
#         n = self.conv3(n)
#         n = torch.relu(n)

#         # Combine and produce the final output
#         combined = torch.cat((q, n), dim=2)
#         combined = combined.view(combined.size(0), -1)
#         output = self.fc(combined)
#         return output




class ConvUpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(ConvUpscaleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

    def forward(self, x, residual):
        x = F.relu(self.conv1(torch.cat((x,residual), dim=1)))
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        residual_upscaled = F.interpolate(residual, size=(x.size(2), x.size(3)), mode='nearest')
        x = torch.cat((x, residual_upscaled), dim=1)
        return x

class ConvOperationLayer(nn.Module):
    def __init__(self, initial_channels, num_blocks, final_out_channels=1):
        super(ConvOperationLayer, self).__init__()
        channels = initial_channels
        self.blocks = nn.ModuleList()
        for i in range(num_blocks-1):
            next_channels = channels // 2
            self.blocks.append(ConvUpscaleBlock(in_channels=channels, out_channels=next_channels, scale_factor=2))       # adding the block of the model
            channels = next_channels + 1                 # adding thiws for orginal map
        self.final_conv = nn.Conv2d(channels, final_out_channels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # x -> from (batch_size, height, width, channels) to (batch_size, channels, height, width) -> reason: conv2d expects the channels first he dimension of the input!!!!!!!!!
        for block in self.blocks:
            residual = x[:, -1:, :, :] 
            x = x[:, :-1, :, :]
            x = block(x, residual)
        x = self.final_conv(x)
        return x




class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class InCTRL(nn.Module):
    def __init__(
            self,
            args,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower_Mul(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.adapter = Adapter(640, 4)
        self.diff_head = TransformerBasicHead(225, 1)
        self.diff_head_ref = TransformerBasicHead(640, 1)

        self.convOperation = ConvOperationLayer(initial_channels=896+1, num_blocks=5, final_out_channels=1)

        for p in self.visual.parameters():
            p.requires_grad = False

        for p in text.parameters():
            p.requires_grad = False

        for p in self.adapter.parameters():
            p.requires_grad = False

        for p in self.diff_head.parameters():
            p.requires_grad = False

        for p in self.diff_head_ref.parameters():
            p.requires_grad = False


        

    def encode_image(self, image, out_layers: list = [7, 9, 11], normalize: bool = False):      # image -> (32, 3, 240, 240); normalize=false
        features = self.visual.forward(image, out_layers)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def peek_localization(self, array_1d, image_tensor):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            array_2d = array_1d.cpu().numpy().reshape((15, 15))
            
            greater_than_point_one = array_2d[array_2d > 0.2]
            if greater_than_point_one.size > 0:
                avg_greater_than_point_one = np.mean(greater_than_point_one)
            else:
                avg_greater_than_point_one = 0

            plt.figure(figsize=(10, 8))
            sns.heatmap(array_2d, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False,vmin=0, vmax=1)
            plt.title(f"15x15 Heatmap of Values (Avg > 0.2: {avg_greater_than_point_one:.2f})")
            plt.xlabel("Column")
            plt.ylabel("Row")
            plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10, 10))
            image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)  # Transpose to HWC format
            plt.imshow(image_np)
            plt.title("Corresponding Image")
            plt.axis('off')
            plt.savefig("corresponding_image.png", dpi=300, bbox_inches='tight')
            plt.close()

    

    def forward(self, inputs: Optional[torch.Tensor] = None, ind = None):
        start1 = time.process_time()
        img = inputs[0].cuda(non_blocking=True)                                                                # image -> (9,32,3,240,240) ; img -> (32,3,240,240)
        shot, b = inputs[-2], inputs[-1]                                            # shot = 8 ; b = 32
        normal_image = inputs[1].cuda(non_blocking=True)                # normal_image -> (8*32=256, 3,240,240)

        token, Fp_list, Fp = self.encode_image(img, normalize=False)                            # token, Fp_list, Fp -> (32,640), (3, 32, 226, 896), (32, 226, 896) -> Fp not used ? (12th layer)
        token_n, Fp_list_n, Fp_n = self.encode_image(normal_image, normalize=False)             # token_n, Fp_list_n, Fp_n -> (256,640), (3,8*32=256,226,896), (8*32=256, 226, 896) -> Fp_n not used ? (12th layer)

        Fp_list = torch.stack(Fp_list)                                                          # Fp_list -> (3, 32, 226, 896)
        Fp_list_n = torch.stack(Fp_list_n)                                                      # Fp_list_n ->  (3,8*32=256,226,896)

        Fp_list = Fp_list[:, :, 1:, :]                                                          # Fp_list -> (3, 32, 225, 896)
        Fp_list_n = Fp_list_n[:, :, 1:, :]                                                      # Fp_list_n ->  (3,8*32=256,225,896)

        Fp_list = Fp_list.reshape(b, 3, 225, -1)                                                # Fp_list -> (32, 3, 225, 896)
        Fp_list_n = Fp_list_n.reshape(b, 3, 225 * shot, -1)                                     # Fp_list_n ->  (32, 3, 225*8=1800, 896)


        # Image level residual
        token_n = token_n.reshape(b, shot, -1)                                                  # token_n -> (32, 8, 640)
        token_ad = self.adapter.forward(token)                                                  # token_ad -> (32, 640)
        token_n = self.adapter.forward(token_n)                                                 # token_n -> (32, 8, 640)
        token_n = torch.mean(token_n, dim=1)                                                    # token_n -> (32, 640)
        token_ref = token_n - token_ad                                      # In context image level residual features # token_ref -> (32, 640)


        # patch level residual
        Fp_list = Fp_list / Fp_list.norm(dim=-1, keepdim=True)
        Fp_list_n = Fp_list_n / Fp_list_n.norm(dim=-1, keepdim=True)
        s = 0.5 * (1 - torch.matmul(Fp_list, Fp_list_n.transpose(-1, -2)))  # s -> (32, 3, 225, 1800)
        am_fp = s.min(dim=-1).values  # am_fp -> (32, 3, 225)
        Fp_map = torch.mean(am_fp, dim=1)  # Fp_map -> (32, 225)
        max_diff_score = Fp_map.max(dim=-1).values  # max_diff_score -> (32)
        patch_ref_map = Fp_map                      # patch_ref_map -> (32, 225)

        
        # zero shot = text image aligned feature extraction
        image_feature = token / token.norm(dim=-1, keepdim=True)  # image_feature -> (32, 640)
        del token, token_ad, Fp, Fp_n                             # freeing up memory -> can I use with?
        del token_n
        del Fp_list_n
        del am_fp
        del s, img, normal_image
        pos_tokens = inputs[2].cuda()  # pos_tokens -> (32*154, 77)
        neg_tokens = inputs[3].cuda()  # neg_tokens -> (32*88, 77)
        start = time.process_time()
        pos_features = self.encode_text(pos_tokens)  # pos_features -> (32*154, 640)
        neg_features = self.encode_text(neg_tokens)  # neg_features -> (32*88, 640)
        print("Time taken by text encoder : ",time.process_time()-start)
        pos_features = pos_features.reshape(b, -1, 640)
        neg_features = neg_features.reshape(b, -1, 640)
        pos_features = torch.mean(pos_features, dim=1, keepdim=True)  # pos_features -> (32, 1, 640)
        neg_features = torch.mean(neg_features, dim=1, keepdim=True)  # neg_features -> (32, 1, 640)
        pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
        neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
        text_features = torch.cat([pos_features, neg_features], dim=1)                # text_features -> (32, 2, 640)
        scores = (100 * image_feature.unsqueeze(1) @ text_features.transpose(-1, -2)).softmax(dim=-1)  # scores -> (32, 1, 2)
        text_score = scores[:, 0, 1]  # text_scores -> (32)


        # Localisation result
        anomaly_localisation_maps = 0
        for i in range(Fp_list.shape[1]): 
            anomaly_localisation_maps += self.convOperation(torch.cat((patch_ref_map.reshape(b,15,15).unsqueeze(-1), Fp_list[:, i, :, :].reshape(b, 15, 15, 896)), dim=-1))
        anomaly_localisation_maps/=Fp_list.shape[1]                                             # taking average as I just want to test -> ()
        # save_images_and_localisation(img, anomaly_localisation_maps, save_dir=f"./TEST/{ind}")





        text_score = text_score.unsqueeze(1)                                       # text_score -> (32,1)      
        img_ref_score = self.diff_head_ref.forward(token_ref)                                   # img_ref_score -> (32, 1)
        # patch_ref_map = torch.stack(patch_ref_map)                                              # patch_ref_map -> (32, 225)
        holistic_map = text_score + img_ref_score + patch_ref_map                               # holistic_map -> (32, 225)
        hl_score = self.diff_head.forward(holistic_map)                                         # hl_score -> (32, 1)

        hl_score = hl_score.squeeze(1)                                                          # hl_score -> (32)
        final_score = (hl_score + max_diff_score) / 2                                                 # final_score -> (32)

        img_ref_score = img_ref_score.squeeze(1)                                                # img_ref_score -> (32)

        print("Time taken by an iteration : ",time.process_time()-start1, " with batch size : ", b)
        return final_score, img_ref_score, anomaly_localisation_maps

class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed

