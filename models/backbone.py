# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
import torch
import torch.nn as nn
import timm


def pad_to_window_size(x, window_size=7):
    """
    将输入 padding 到 window_size 的倍数
    """
    B, C, H, W = x.shape

    # 计算需要 padding 到的尺寸
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    return x, pad_h, pad_w


class SwinBackbone(nn.Module):
    def __init__(self, name="swin_tiny_patch4_window7_224", out_dim=256, pretrained=True):
        super().__init__()

        # 使用固定尺寸创建模型，然后我们会在forward中处理动态尺寸
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            img_size=224  # 先用固定尺寸创建
        )

        in_dim = self.backbone.feature_info[-1]['num_chs']
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.num_channels = out_dim

        # 保存原始配置
        self.patch_size = 4
        self.window_size = 7

    def _update_all_img_sizes(self, H, W):
        """递归更新所有模块的img_size相关参数"""

        def update_module(module):
            # 更新patch_embed
            if hasattr(module, 'img_size'):
                module.img_size = (H, W)
            if hasattr(module, 'grid_size'):
                grid_h = H // self.patch_size
                grid_w = W // self.patch_size
                module.grid_size = (grid_h, grid_w)
            if hasattr(module, 'num_patches'):
                module.num_patches = (H // self.patch_size) * (W // self.patch_size)

            # 更新position embedding相关
            if hasattr(module, 'absolute_pos_embed'):
                # 动态调整绝对位置编码
                old_pos_embed = module.absolute_pos_embed
                if old_pos_embed is not None:
                    new_num_patches = (H // self.patch_size) * (W // self.patch_size)
                    if old_pos_embed.shape[1] != new_num_patches + 1:  # +1 for cls token
                        # 插值调整位置编码
                        old_size = int((old_pos_embed.shape[1] - 1) ** 0.5)
                        new_size_h = H // self.patch_size
                        new_size_w = W // self.patch_size

                        cls_pos_embed = old_pos_embed[:, 0:1, :]
                        pos_embed = old_pos_embed[:, 1:, :].reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
                        pos_embed = F.interpolate(pos_embed, size=(new_size_h, new_size_w), mode='bicubic',
                                                  align_corners=False)
                        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_size_h * new_size_w, -1)
                        module.absolute_pos_embed = nn.Parameter(torch.cat([cls_pos_embed, pos_embed], dim=1))

            # 递归处理子模块
            for child in module.children():
                update_module(child)

        update_module(self.backbone)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None

        B, C, H, W = x.shape

        # 解决方案：调整到窗口大小的倍数，并且是patch_size的倍数
        # 确保输入尺寸符合Swin Transformer的要求

        # 计算调整后的尺寸
        def get_valid_size(size):
            # 必须是patch_size的倍数
            size = ((size + self.patch_size - 1) // self.patch_size) * self.patch_size
            # 在patch embedding后，feature map的尺寸必须是window_size的倍数
            feat_size = size // self.patch_size
            if feat_size % self.window_size != 0:
                feat_size = ((feat_size + self.window_size - 1) // self.window_size) * self.window_size
                size = feat_size * self.patch_size
            return size

        # 限制最大尺寸避免内存爆炸
        max_size = 800
        if H > max_size or W > max_size:
            scale = min(max_size / H, max_size / W)
            H = int(H * scale)
            W = int(W * scale)

        new_h = get_valid_size(H)
        new_w = get_valid_size(W)

        # 进一步限制，避免太大
        if new_h > max_size:
            new_h = max_size
            new_h = get_valid_size(new_h)
        if new_w > max_size:
            new_w = max_size
            new_w = get_valid_size(new_w)

        #print(f"调整尺寸从 {x.shape[-2:]} 到 {(new_h, new_w)}")

        # 调整输入
        if H != new_h or W != new_w:
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            mask = F.interpolate(mask[None].float(), size=(new_h, new_w), mode='nearest').bool()[0]

        # 彻底更新所有相关的尺寸参数
        self._update_all_img_sizes(new_h, new_w)

        try:
            features = self.backbone(x)
            x = features[-1]
        except RuntimeError as e:
            if "shape" in str(e) and "invalid for input of size" in str(e):
                #print(f"仍有形状错误: {e}")
                #print("尝试更保守的尺寸...")

                # 使用更小的尺寸
                safe_h = 224
                safe_w = 224

                x = F.interpolate(tensor_list.tensors, size=(safe_h, safe_w), mode='bilinear', align_corners=False)
                mask = F.interpolate(mask[None].float(), size=(safe_h, safe_w), mode='nearest').bool()[0]

                self._update_all_img_sizes(safe_h, safe_w)
                features = self.backbone(x)
                x = features[-1]
            else:
                raise e

        # 处理输出维度
        if len(x.shape) == 3:
            # 从[B, H*W, C]转换为[B, C, H, W]
            B_out, HW, C_out = x.shape
            H_out = W_out = int(HW ** 0.5)
            x = x.transpose(1, 2).view(B_out, C_out, H_out, W_out)
        elif len(x.shape) == 4 and x.shape[1] != self.proj.in_channels:
            # 从[B, H, W, C]转换为[B, C, H, W]
            if x.shape[-1] == self.proj.in_channels:
                x = x.permute(0, 3, 1, 2)

        # 投影
        x = self.proj(x)

        # 调整mask
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask[None].float(), size=x.shape[-2:], mode='nearest').bool()[0]

        return {'0': NestedTensor(x, mask)}




def build_swin_backbone(args):
    if args.backbone.startswith('swin'):
        backbone_name = {
            'swin': 'swin_tiny_patch4_window7_224',
            'swin_tiny': 'swin_tiny_patch4_window7_224',
            'swin_small': 'swin_small_patch4_window7_224',
            'swin_base': 'swin_base_patch4_window7_224'
        }.get(args.backbone, 'swin_tiny_patch4_window7_224')

        # 如果主方案有问题，取消注释下面这行使用保守方案
        # return SwinBackboneConservative(name=backbone_name)
        return SwinBackbone(name=backbone_name)
    else:
        from .backbone import build_backbone
        return build_backbone(args)

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    from .position_encoding import build_position_encoding

    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks

    #backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    if args.backbone.startswith('resnet'):
        # 原始 ResNet
        from .backbone import Backbone
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    elif args.backbone.startswith('swin'):
        # Swin Transformer
        from .backbone import SwinBackbone
        backbone = SwinBackbone(name=args.backbone)
    else:
        raise NotImplementedError(f"当前只支持 resnet 或 swin 系列，收到: {args.backbone}")

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
