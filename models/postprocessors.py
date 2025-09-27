# models/postprocessors.py

import torch
import torch.nn.functional as F

def build_postprocessors(args):
    """
    返回 postprocessors 字典，用于 evaluate 阶段。
    即使只做 bbox，也不能返回 None。
    """
    postprocessors = {}

    # ------------------------------
    # 1️⃣ 目标检测 bbox
    # ------------------------------
    def postprocess_bbox(outputs, target_sizes):
        """
        outputs: dict, 包含 'pred_logits' 和 'pred_boxes'
        target_sizes: 原图尺寸，用于将 normalized bbox 转回像素坐标
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']  # [batch, num_queries, ...]
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)  # 去掉 no-object 类

        # 将 bbox 从 [0,1] 转换到原图尺寸
        img_h, img_w = target_sizes.unbind(1)
        img_h = img_h[:, None]
        img_w = img_w[:, None]
        # out_bbox 形状: [batch, num_queries, 4], xywh normalized
        bboxes = out_bbox.clone()
        bboxes[..., 0] *= img_w  # cx
        bboxes[..., 1] *= img_h  # cy
        bboxes[..., 2] *= img_w  # w
        bboxes[..., 3] *= img_h  # h

        # 返回 list[dict] 每张图一个 dict
        return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, bboxes)]

    postprocessors['bbox'] = postprocess_bbox

    # ------------------------------
    # 2️⃣ 分割 segm (可选)
    # ------------------------------
    if args.masks:
        def postprocess_segmentation(outputs, target_sizes):
            """
            outputs: dict, 包含 'pred_masks'
            target_sizes: 原图尺寸，用于 resize
            """
            masks = outputs['pred_masks']  # [batch, num_queries, H, W]
            # resize mask 到原图大小
            # 这里简单返回原尺寸 mask，evaluate 时再处理
            return masks
        postprocessors['segm'] = postprocess_segmentation

    return postprocessors
