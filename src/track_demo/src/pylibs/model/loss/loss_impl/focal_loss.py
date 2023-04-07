# -*- coding: utf-8 -*

import torch

from ...common_opr.common_loss import sigmoid_focal_loss_jit
from ...module_base import ModuleBase
from ..loss_base import TRACK_LOSSES



@TRACK_LOSSES.register
class FocalLoss(ModuleBase):

    default_hyper_params = dict(
        name="focal_loss",
        background=0,
        ignore_label=-1,
        weight=1.0,
        alpha=0.5,
        gamma=0.0,
    )

    def __init__(self, ):
        super().__init__()
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))

    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def forward(self, pred_data, target_data):
        r"""
        Focal loss
        :param pred: shape=(B, HW, C), classification logits (BEFORE Sigmoid)
        :param label: shape=(B, HW)
        """
        r"""
        Focal loss
        Arguments
        ---------
        pred: torch.Tensor
            classification logits (BEFORE Sigmoid)
            format: (B, HW)
        label: torch.Tensor
            training label
            format: (B, HW)

        Returns
        -------
        torch.Tensor
            scalar loss
            format: (,)
        """
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
        from torch.nn.functional import l1_loss

        l1_loss = l1_loss(pred_boxes_vec, gt_boxes_vec)
        extra = dict()
        return 5*l1_loss, extra
