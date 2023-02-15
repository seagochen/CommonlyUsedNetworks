import torch
import torch.nn as nn


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Filter out overlapping bounding boxes based on their confidence scores and IoU.

    Args:
        prediction: tensor of shape (batch_size, num_anchors, num_classes + 5, grid_size, grid_size)
        conf_thres: threshold for objectness confidence score
        nms_thres: threshold for non-maximum suppression based on IoU

    Returns:
        detections: tensor of shape (batch_size, max_detections, 6), where each detection is of the form
            (x1, y1, x2, y2, objectness_score, class_score, class_id)
    """

    # Get the boxes with objectness score greater than the threshold
    conf_mask = (prediction[..., 4] >= conf_thres).float()

    # Apply sigmoid to the class scores
    prediction[..., 5:] = torch.sigmoid(prediction[..., 5:])

    # Get the scores of the highest class for each box
    class_scores, class_ids = torch.max(prediction[..., 5:], dim=-1)
    class_scores = class_scores * conf_mask

    # Compute the coordinates of the boxes
    xywh = prediction[..., :4]
    x1y1 = xywh[..., :2] - xywh[..., 2:] / 2
    x2y2 = xywh[..., :2] + xywh[..., 2:] / 2
    boxes = torch.cat((x1y1, x2y2), dim=-1)

    # Perform non-maximum suppression
    detections = []
    for i in range(prediction.shape[0]):
        boxes_i = boxes[i]
        scores_i = class_scores[i]
        class_ids_i = class_ids[i]

        keep = []
        while scores_i.numel() > 0:
            max_idx = torch.argmax(scores_i)
            keep.append(max_idx)

            if len(scores_i) == 1:
                break

            ious = bbox_iou(boxes_i[max_idx].unsqueeze(0), boxes_i)
            scores_i = scores_i[ious <= nms_thres]
            class_ids_i = class_ids_i[ious <= nms_thres]
            boxes_i = boxes_i[ious <= nms_thres]

        detections_i = torch.cat((
            boxes[keep],
            class_scores[i][keep].unsqueeze(-1),
            class_ids[i][keep].unsqueeze(-1).float()
        ), dim=-1)

        num_padding = prediction.shape[1] - detections_i.shape[0]
        if num_padding > 0:
            detections_i = torch.cat((
                detections_i,
                torch.zeros((num_padding, detections_i.shape[1]), device=detections_i.device)
            ), dim=0)

        detections.append(detections_i)

    detections = torch.stack(detections, dim=0)

    # Return the detections
    return detections


def bbox_iou(box1, box2):
    """
    Compute the intersection over union (IoU) of two sets of bounding boxes.

    Args:
        box1: tensor of shape (n, 4) representing bounding boxes in the format (x1, y1, x2, y2)
        box2: tensor of shape (m, 4) representing bounding boxes in the format (x1, y1, x2, y2)

    Returns:
        iou: tensor of shape (n, m) representing the IoU of each pair of boxes
    """
    # Compute the coordinates of the intersection rectangle
    x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
    y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
    x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
    y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))

    # Compute the area of the intersection rectangle
    intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Compute the areas of each bounding box
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # Compute the union area
    union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - intersection_area

    # Compute the intersection over union (IoU)
    iou = intersection_area / union_area

    return iou
