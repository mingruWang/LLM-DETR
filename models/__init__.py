# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
'''


from .detr import build


def build_model(args):
    return build(args)
'''
from .backbone import build_swin_backbone
from .detr import DETR
from .matcher import build_matcher
from .detr import SetCriterion
from .detr import DETR, SetCriterion
from .backbone import build_swin_backbone
from .transformer import build_transformer
from .postprocessors import build_postprocessors

import torch
import torch.nn as nn

from .backbone import build_swin_backbone   # Swin backbone
from .transformer import build_transformer  # 返回 Transformer 实例
from .detr import DETR, SetCriterion
from .matcher import build_matcher

from .backbone import build_backbone
from .detr import DETR
from .matcher import HungarianMatcher
from .postprocessors import build_postprocessors
from .transformer import build_transformer

def build_model(args):
    """
    构建 DETR 模型，包括 backbone、transformer、matcher、criterion
    """
    # 1. Backbone

    backbone = build_backbone(args)  # 返回 nn.Module，必须包含 num_channels


    # 2. Transformer
    from .transformer import build_transformer
    transformer = build_transformer(args)

    # 3. 构建 DETR 模型
    num_classes = getattr(args, "num_classes", 91)  # COCO 默认 91 类
    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss
    )

    # 4. Matcher
    matcher = HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou
    )

    # 5. Criterion
    weight_dict = {
        'loss_ce': 1,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef
    }
    if args.masks:
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ['masks']

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses
    )
    criterion.to(args.device)

    # 6. Postprocessors
    postprocessors = build_postprocessors(args)

    return model, criterion, postprocessors