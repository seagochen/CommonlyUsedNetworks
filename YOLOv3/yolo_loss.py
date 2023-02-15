import torch
import torch.nn as nn

from .utils import bbox_iou

class YOLOv3Loss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOv3Loss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thresh = 0.5
        self.img_size = img_size
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets):
        b, _, h, w = predictions[0].size()
        stride = self.img_size / h
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        anchor_w = torch.tensor([an_w for an_w, _ in scaled_anchors], dtype=torch.float32, device=predictions[0].device)
        anchor_h = torch.tensor([an_h for _, an_h in scaled_anchors], dtype=torch.float32, device=predictions[0].device)
        grid_x, grid_y = torch.meshgrid([torch.arange(w, device=predictions[0].device), torch.arange(h, device=predictions[0].device)])
        x_offset = grid_x.view((1, 1, h, w)).expand((1, self.num_anchors, h, w)).float()
        y_offset = grid_y.view((1, 1, h, w)).expand((1, self.num_anchors, h, w)).float()
        scaled_anchors_w = anchor_w.view((1, self.num_anchors, 1, 1)).expand((1, self.num_anchors, h, w))
        scaled_anchors_h = anchor_h.view((1, self.num_anchors, 1, 1)).expand((1, self.num_anchors, h, w))
        bbox_preds = torch.cat([p[:, : self.num_anchors * 4, :, :].reshape(b, self.num_anchors, 4, h, w).permute(0, 1, 3, 4, 2) for p in predictions], dim=1)
        obj_preds = torch.cat([p[:, self.num_anchors * 4: self.num_anchors * 5, :, :] for p in predictions], dim=1)
        class_preds = torch.cat([p[:, self.num_anchors * 5 :, :, :] for p in predictions], dim=1)
        tgt_mask = targets[..., 4] > 0
        tgt_bbox = targets[tgt_mask, :4].view(-1, 4)
        tgt_obj = targets[..., 4][tgt_mask]
        tgt_class = targets[..., 5][tgt_mask]
        tgt_bbox[..., [0, 2]] = tgt_bbox[..., [0, 2]] * w - x_offset[tgt_mask]
        tgt_bbox[..., [1, 3]] = tgt_bbox[..., [1, 3]] * h - y_offset[tgt_mask]
        tgt_bbox[..., [0, 2]] /= scaled_anchors_w[tgt_mask]
        tgt_bbox[..., [1, 3]] /= scaled_anchors_h[tgt_mask]
        pred_bbox = bbox_preds[tgt_mask]
        iou = bbox_iou(pred_bbox, tgt_bbox)
        best_iou, best_box = iou.max(1)
        obj_mask = best_iou > self.ignore_thresh
        noobj_mask = best_iou < self.ignore_thresh
        obj_mask = obj_mask.unsqueeze(-1).expand_as(obj_preds)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(obj_preds)
        # Compute objectness loss
        obj_loss = self.bce_loss(obj_preds[obj_mask], tgt_obj[obj_mask])
        noobj_loss = self.bce_loss(obj_preds[noobj_mask], tgt_obj[noobj_mask])
        obj_weight = obj_mask.size(0) / (obj_mask.sum() + 1e-6)
        noobj_weight = noobj_mask.size(0) / (noobj_mask.sum() + 1e-6)
        obj_loss = obj_weight * obj_loss.mean()
        noobj_loss = noobj_weight * noobj_loss.mean()
        # Compute localization loss
        tgt_bbox = torch.zeros_like(bbox_preds)
        tgt_bbox[tgt_mask, best_box, :, :, :] = tgt_bbox.new_tensor(tgt_bbox)
        box_loss = self.mse_loss(bbox_preds[obj_mask], tgt_bbox[obj_mask])
        box_loss = box_loss.mean()
        # Compute classification loss
        class_mask = torch.zeros_like(class_preds)
        tgt_class = tgt_class.to(torch.int64)
        class_mask[tgt_mask, tgt_class] = 1.0
        class_loss = self.bce_loss(class_preds[obj_mask], class_mask[obj_mask])
        class_weight = tgt_mask.size(0) / (tgt_mask.sum() + 1e-6)
        class_loss = class_weight * class_loss.mean()
        # Compute total loss
        loss = obj_loss + noobj_loss + box_loss + class_loss
        return loss



