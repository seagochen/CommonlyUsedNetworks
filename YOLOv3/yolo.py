import torch.nn as nn


class DarknetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DarknetConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = DarknetConv(in_channels, in_channels // 2, 1)
        self.conv2 = DarknetConv(in_channels // 2, in_channels, 3, padding=1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return x


import torch
import torch.nn as nn
from utils import bbox_iou, non_max_suppression

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thresh = 0.5
        self.img_size = img_size

    def forward(self, x, targets=None):
        n, _, h, w = x.shape

        x = x.view(n, self.num_anchors, self.num_classes + 5, h, w)
        x = x.permute(0, 1, 3, 4, 2)

        bbox_xy = torch.sigmoid(x[..., :2])
        bbox_wh = torch.exp(x[..., 2:4])
        obj_conf = torch.sigmoid(x[..., 4])
        class_conf = torch.sigmoid(x[..., 5:])

        grid_x = torch.arange(w, dtype=torch.float32, device=x.device).repeat(h, 1).view([1, 1, h, w]).repeat([n, self.num_anchors, 1, 1])
        grid_y = torch.arange(h, dtype=torch.float32, device=x.device).repeat(w, 1).t().view([1, 1, h, w]).repeat([n, self.num_anchors, 1, 1])

        anchor_w = torch.tensor([an_w for an_w, _ in self.anchors], dtype=torch.float32, device=x.device)
        anchor_h = torch.tensor([an_h for _, an_h in self.anchors], dtype=torch.float32, device=x.device)

        pred_boxes = torch.zeros_like(x[..., :4])
        pred_boxes[..., 0] = bbox_xy[..., 0] + grid_x
        pred_boxes[..., 1] = bbox_xy[..., 1] + grid_y
        pred_boxes[..., 2] = bbox_wh[..., 0] * anchor_w.view([1, self.num_anchors, 1, 1])
        pred_boxes[..., 3] = bbox_wh[..., 1] * anchor_h.view([1, self.num_anchors, 1, 1])

        if targets is not None:
            obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.build_targets(pred_boxes, targets, self.ignore_thresh)

            loss_x = nn.BCEWithLogitsLoss(reduction='sum')(bbox_xy[obj_mask], tx[obj_mask])
            loss_y = nn.BCEWithLogitsLoss(reduction='sum')(bbox_xy[obj_mask], ty[obj_mask])
            loss_w = nn.MSELoss(reduction='sum')(bbox_wh[obj_mask], tw[obj_mask])
            loss_h = nn.MSELoss(reduction='sum')(bbox_wh[obj_mask], th[obj_mask])
            loss_conf_obj = nn.BCEWithLogitsLoss(reduction='sum')(obj_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = nn.BCEWithLogitsLoss(reduction='sum')(obj_conf[noobj_mask], tconf[noobj_mask])
            loss_cls = nn.BCEWithLogitsLoss(reduction='sum')(class_conf[obj_mask], tcls[obj_mask])

            loss = loss_x + loss_y + loss_w + loss_h + loss_conf_obj + 100.0 * loss_conf_noobj + loss_cls

            return loss

        else:
            output = torch.cat((pred_boxes.view(n, -1, 4) * self.img_size, obj_conf.view(n, -1, 1), class_conf.view(n, -1, self.num_classes)), -1)
            if targets is not None:
                boxes = targets[:, 2:6] * torch.tensor([w, h, w, h], device=x.device)
                boxes[:, :2] -= boxes[:, 2:] / 2

                iou_scores = bbox_iou(pred_boxes.detach().view(-1, 4), boxes.detach())
                iou_scores = iou_scores.view(n, self.num_anchors, h, w)

                best_iou, best_box = torch.max(iou_scores, dim=1, keepdim=True)
                obj_mask = (best_iou > self.ignore_thresh).float().unsqueeze(-1)
                noobj_mask = (best_iou <= self.ignore_thresh).float().unsqueeze(-1)

                # Set target confidence
                tconf = obj_mask.clone()
                tconf[noobj_mask == 1] = 0

                # Set target class probabilities
                tcls = targets[:, 1].long()

                # One-hot encoding
                tcls_onehot = torch.zeros(n, self.num_anchors, 1, self.num_classes, device=x.device)
                tcls_onehot.scatter_(3, tcls.view(n, 1, 1, 1), 1)

                # Compute loss for class probabilities
                loss_cls = nn.BCEWithLogitsLoss(reduction='none')(class_conf, tcls_onehot)
                loss_cls = (loss_cls * obj_mask).sum()

                # Compute loss for objectness confidence
                loss_conf_obj = nn.BCEWithLogitsLoss(reduction='none')(obj_conf, tconf)
                loss_conf_obj = (loss_conf_obj * obj_mask).sum()

                # Compute loss for no objectness confidence
                loss_conf_noobj = nn.BCEWithLogitsLoss(reduction='none')(obj_conf, tconf)
                loss_conf_noobj = (loss_conf_noobj * noobj_mask).sum()

                # Compute loss for x-coordinates
                loss_x = self.mse_loss(bbox_xy[..., 0], tx)
                loss_x = (loss_x * obj_mask).sum()

                # Compute loss for y-coordinates
                loss_y = self.mse_loss(bbox_xy[..., 1], ty)
                loss_y = (loss_y * obj_mask).sum()

                # Compute loss for widths
                loss_w = self.mse_loss(bbox_wh[..., 0], torch.log(tw / anchor_w.view([1, self.num_anchors, 1, 1])))
                loss_w = (loss_w * obj_mask).sum()

                # Compute loss for heights
                loss_h = self.mse_loss(bbox_wh[..., 1], torch.log(th / anchor_h.view([1, self.num_anchors, 1, 1])))
                loss_h = (loss_h * obj_mask).sum()

                # Total loss
                loss = loss_x + loss_y + loss_w + loss_h + loss_conf_obj + 100.0 * loss_conf_noobj + loss_cls

                return output, loss

            else:
                output[..., :4] *= self.img_size

                # Apply non-maximum suppression to filter out overlapping boxes
                detections = non_max_suppression(output.view(n, -1, 6), conf_thres=0.5, nms_thres=0.4)

                return detections




class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv3, self).__init__()

        self.layer1 = nn.Sequential(
            DarknetConv(3, 32, 3, padding=1),
            DarknetConv(32, 64, 3, stride=2, padding=1),
            ResidualBlock(64, 32),
            DarknetConv(32, 128, 3, stride=2, padding=1),
            ResidualBlock(128, 64),
            ResidualBlock(64, 128),
            DarknetConv(128, 256, 3, stride=2, padding=1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 128),
            ResidualBlock(128, 256),
            DarknetConv(256, 512, 3, stride=2, padding=1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 512),
            ResidualBlock(512, 256),
            ResidualBlock(256, 512),
            ResidualBlock(512, 256),
            ResidualBlock(256, 512),
            ResidualBlock(512, 256),
            ResidualBlock(256, 512),
            DarknetConv(512, 1024, 3, stride=2, padding=1),
            ResidualBlock(1024, 512),
            ResidualBlock(512, 1024),
            ResidualBlock(1024, 512),
            ResidualBlock(512, 1024),
            DarknetConv(1024, 512, 1),
            DarknetConv(512, 1024, 3),
            DarknetConv(1024, 512, 1),
            DarknetConv(512, 1024, 3),
            DarknetConv(1024, 512, 1),
            DarknetConv(512, 1024, 3)
        )

        self.layer2 = nn.Sequential(
            DarknetConv(512, 256, 1),
            nn.Upsample(scale_factor=2),
            DarknetConv(256+512, 256, 1),
            DarknetConv(256, 512, 3),
            DarknetConv(512, 256, 1),
            DarknetConv(256, 512, 3),
            DarknetConv(512, 256, 1),
            DarknetConv(256, 512, 3),
        )

        self.layer3 = nn.Sequential(
            DarknetConv(512, 128, 1),
            nn.Upsample(scale_factor=2),
            DarknetConv(128+256, 128, 1),
            DarknetConv(128, 256, 3),
            DarknetConv(256, 128, 1),
            DarknetConv(128, 256, 3),
        )

        self.layer4 = nn.Sequential(
            DarknetConv(256, 64, 1),
            nn.Upsample(scale_factor=2),
            DarknetConv(64+128, 64, 1),
            DarknetConv(64, 128, 3),
            DarknetConv(128, 64, 1),
            DarknetConv(64, 128, 3),
            DarknetConv(128, 64, 1),
            DarknetConv(64, 128, 3),
        )

        self.layer5 = nn.Sequential(
            DarknetConv(128, 32, 1),
            DarknetConv(32, 64, 3),
            DarknetConv(64, 32, 1),
            DarknetConv(32, 64, 3),
            DarknetConv(64, num_classes+5, 1, bn=False, act=False)
        )

        self.yolo_layer1 = YOLOLayer(anchor_mask=[6,7,8], num_classes=num_classes)
        self.yolo_layer2 = YOLOLayer(anchor_mask=[3,4,5], num_classes=num_classes)
        self.yolo_layer3 = YOLOLayer(anchor_mask=[0,1,2], num_classes=num_classes)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        y1, y2, y3 = self.yolo_layer1(x5), self.yolo_layer2(torch.cat([x4, y1], dim=1)), self.yolo_layer3(torch.cat([x3, y2], dim=1))
        return [y1, y2, y3]