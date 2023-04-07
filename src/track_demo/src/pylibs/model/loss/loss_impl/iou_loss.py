# -*- coding: utf-8 -*
import numpy as np

import torch

from torchvision.ops.boxes import box_area

from ...module_base import ModuleBase
from ..loss_base import TRACK_LOSSES
from .utils import SafeLog

eps = np.finfo(np.float32).tiny


@TRACK_LOSSES.register
class IOULoss(ModuleBase):

    default_hyper_params = dict(
        name="iou_loss",
        background=0,
        ignore_label=-1,
        weight=1.0,
    )

    def __init__(self):
        super().__init__()
        self.safelog = SafeLog()
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))
        self.register_buffer("t_zero", torch.tensor(0., requires_grad=False))

    def update_params(self):
        self.background = self._hyper_params["background"]
        self.ignore_label = self._hyper_params["ignore_label"]
        self.weight = self._hyper_params["weight"]

    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def box_xywh_to_xyxy(self,x):
        x1, y1, w, h = x.unbind(-1)
        b = [x1, y1, x1 + w, y1 + h]
        return torch.stack(b, dim=-1)

    def box_xyxy_to_xywh(self,x):
        x1, y1, x2, y2 = x.unbind(-1)
        b = [x1, y1, x2 - x1, y2 - y1]
        return torch.stack(b, dim=-1)

    def box_xyxy_to_cxcywh(self,x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    # modified from torchvision to also return the union
    '''Note that this function only supports shape (N,4)'''

    def box_iou(self,boxes1, boxes2):
        """

        :param boxes1: (N, 4) (x1,y1,x2,y2)
        :param boxes2: (N, 4) (x1,y1,x2,y2)
        :return:
        """
        area1 = box_area(boxes1)  # (N,)
        area2 = box_area(boxes2)  # (N,)

        lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

        wh = (rb - lt).clamp(min=0)  # (N,2)
        inter = wh[:, 0] * wh[:, 1]  # (N,)

        union = area1 + area2 - inter

        iou = inter / union
        return iou, union

    '''Note that this implementation is different from DETR's'''

    def generalized_box_iou(self,boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/

        The boxes should be in [x0, y0, x1, y1] format

        boxes1: (N, 4)
        boxes2: (N, 4)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        # try:
        # print((boxes1[:, 2:] >= boxes1[:, :2]).all() and (boxes2[:, 2:] >= boxes2[:, :2]).all())

        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        iou, union = self.box_iou(boxes1, boxes2)  # (N,)

        lt = torch.min(boxes1[:, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # (N,2)
        area = wh[:, 0] * wh[:, 1]  # (N,)

        return iou - (area - union) / area, iou

    def giou_loss(self,boxes1, boxes2):
        """

        :param boxes1: (N, 4) (x1,y1,x2,y2)
        :param boxes2: (N, 4) (x1,y1,x2,y2)
        :return:
        """
        giou, iou = self.generalized_box_iou(boxes1, boxes2)
        return (1 - giou).mean(), iou

    def clip_box(self,box: list, H, W, margin=0):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        x1 = min(max(0, x1), W - margin)
        x2 = min(max(margin, x2), W)
        y1 = min(max(0, y1), H - margin)
        y2 = min(max(margin, y2), H)
        w = max(margin, x2 - x1)
        h = max(margin, y2 - y1)
        return [x1, y1, w, h]

    def forward(self, pred_data, target_data):
        pred_boxes = pred_data["box_pred"]
        if torch.isnan(pred_boxes).any():
            print(pred_boxes,target_data["bbox_x"])
            raise ValueError("Network outputs is NAN! Stop Training")
        gt = target_data["bbox_x"]/303
        # gtx1,gty1,gtx2,gty2 = gt[:,]/303
        # gt1 =
        # gt_box = torch.cat()
        pred_boxes_vec = self.box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        # print(pred_boxes_vec)
        # print(pred_boxes_vec)
        gt_boxes_vec = gt[:, None, :].repeat((1, 1, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
        # print(gt_boxes_vec)
        try:
            giou_loss, iou = self.giou_loss(pred_boxes_vec, gt_boxes_vec)
        except AssertionError:
            giou_loss, iou = torch.tensor(0.0).cuda(pred_boxes_vec.device), torch.tensor(0.0).cuda(pred_boxes_vec.device)

        extra = dict(iou=iou.mean())
        loss = giou_loss
        return loss, extra


if __name__ == '__main__':
    B = 16
    HW = 17 * 17
    pred_cls = pred_ctr = torch.tensor(
        np.random.rand(B, HW, 1).astype(np.float32))
    pred_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))

    gt_cls = torch.tensor(np.random.randint(2, size=(B, HW, 1)),
                          dtype=torch.int8)
    gt_ctr = torch.tensor(np.random.rand(B, HW, 1).astype(np.float32))
    gt_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))

    criterion_cls = SigmoidCrossEntropyRetina()
    loss_cls = criterion_cls(pred_cls, gt_cls)

    criterion_ctr = SigmoidCrossEntropyCenterness()
    loss_ctr = criterion_ctr(pred_ctr, gt_ctr, gt_cls)

    criterion_reg = IOULoss()
    loss_reg = criterion_reg(pred_reg, gt_reg, gt_cls)

    from IPython import embed
    embed()
