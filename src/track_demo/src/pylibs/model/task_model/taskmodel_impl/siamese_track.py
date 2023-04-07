# -*- coding: utf-8 -*

from loguru import logger
from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf
from einops import rearrange, repeat
import time
from model.common_opr.common_block import (conv_bn_relu,
                                                        xcorr_depthwise)
from model.module_base import ModuleBase
from model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)
from torchvision import models
from functools import partial
from torchvision.ops.misc import ConvNormActivation
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig
from torch import nn, Tensor
from typing import Optional, List
import numpy as np


torch.set_printoptions(precision=8)

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 取掉model的后1层
        self.conv0 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_layer = nn.Sequential(*list(model.children())[1:-4])
        # self.Linear_layer = nn.Linear(512, 2)  # 加上一层参数修改好的全连接层

    def forward(self, x):
        x = self.conv0(x)
        x = self.resnet_layer(x)
        # x = x.view(x.size(0), -1)
        # x = self.Linear_layer(x)
        return x

class Net_vgg(nn.Module):
    def __init__(self, model):
        super(Net_vgg, self).__init__()
        # 取掉model的后1层
        self.resnet_layer = nn.Sequential(*list(model.features.children())[:-14])
        # self.Linear_layer = nn.Linear(512, 2)  # 加上一层参数修改好的全连接层

    def forward(self, x):
        x = self.resnet_layer(x)
        # x = x.view(x.size(0), -1)
        # x = self.Linear_layer(x)
        return x

class Blocks(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        block = InvertedResidual
        reduce_divider = 1
        dilation = 1
        bneck_conf = partial(InvertedResidualConfig, width_mult=1.0)
        confs = [bneck_conf(64, 5, 128, 64, True, "RE", 2, 1),
                #  bneck_conf(16, 3, 48, 24, True, "RE", 2, 1),  # C1
                #  bneck_conf(24, 3, 192, 48, True, "RE", 1, 1),
                #  bneck_conf(64, 5, 192, 64, True, "RE", 1, 1),  # C2
                #  bneck_conf(48, 5, 256, 64, True, "RE", 1, 1),
                 bneck_conf(64, 3, 256, 128, True, "RE", 2, 1),
                 ]
        # confs = [bneck_conf(128, 5, 240, 128, True, "HS", 1, 1),
        #          bneck_conf(128, 5, 240, 128, True, "HS", 1, 1),
        #          bneck_conf(128, 5, 240, 128, True, "HS", 1, 1),]
        layers: List[nn.Module] = []
        firstconv_output_channels = confs[0].input_channels
        layers.append(
            ConvNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU,
            )
        )
        for cnf in confs:
            layers.append(block(cnf, norm_layer))
        self.features = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.features(x)

def conv_bn_re(inch, outch, ksize=(3, 3), stride=(1, 1), pad=(1, 1), bias=False):
    return nn.Sequential(
        nn.Conv2d(inch, outch, kernel_size=ksize,
                  stride=stride, padding=pad, bias=bias),
        nn.BatchNorm2d(outch, eps=1e-05, momentum=0.1,
                       affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
    )


def conv_bn(inch, outch, ksize=(3, 3), stride=(1, 1), pad=(1, 1), bias=False):
    return nn.Sequential(
        nn.Conv2d(inch, outch, kernel_size=ksize,
                  stride=stride, padding=pad, bias=bias),
        nn.BatchNorm2d(outch, eps=1e-05, momentum=0.1,
                       affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
    )

def down(inch, outch, ksize=(3, 3), stride=(1, 1)):
    return nn.Sequential(
        nn.Conv2d(inch,outch,kernel_size=ksize,stride= stride, bias=False),
        nn.BatchNorm2d(outch, eps=1e-05, momentum=0.1,
                       affine=True, track_running_stats=True),
    )


class My_Res(nn.Module):
    def __init__(self) -> None:
        super(My_Res, self).__init__()
        self.conv0 = conv_bn_re(3, 64, (7, 7), (2, 2), (3, 3), False)
        self.maxpool0 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv1 = conv_bn_re(64, 64)
        self.conv2 = conv_bn(64, 64)
        self.conv3 = conv_bn_re(64, 64)
        self.conv4 = conv_bn(64, 64)
        self.conv5 = conv_bn_re(64, 128, stride=(2, 2))
        self.conv6 = conv_bn(128, 128)
        self.down0 = down(64, 128, (1, 1), stride=(2, 2))
        self.conv7 = conv_bn_re(128, 128)
        self.conv8 = conv_bn(128, 128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.maxpool0(x)
        i1 = x
        x = self.conv1(x)
        x = self.conv2(x)
        x =x + i1
        x = self.relu(x)
        i2 = x
        x = self.conv3(x)
        x = self.conv4(x)
        x =x + i2
        x = self.relu(x)
        i4 = x
        i3 = x
        x = self.conv5(x)
        x = self.conv6(x)
        i3 = self.down0(i4)
        x = x + i3
        x = self.relu(x)
        i4 = x
        x = self.conv7(x)
        x = self.conv8(x)
        x = x + i4
        x = self.relu(x)
        return x


@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class SiamTrack(ModuleBase):
    r"""
    SiamTrack model for tracking

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    head_width: int
        feature width in head structure
    """

    default_hyper_params = dict(pretrain_model_path="",
                                head_width=128,
                                conv_weight_std=0.01,
                                neck_conv_bias=[True, True, True, True],
                                corr_fea_output=False,
                                trt_mode=False,
                                trt_fea_model_path="",
                                trt_track_model_path="",
                                amp=False)

    support_phases = ["train", "feature", "track", "freeze_track_fea"]

    def __init__(self, backbone, head, loss=None):
        super(SiamTrack, self).__init__()
        self.head = head
        self.loss = loss
        # num_queries = 1
        self.mean = [0.485,0.456,0.406]
        self.std = [0.229,0.224,0.225]
        # hidden_dim = 256
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # self.attention = Counter_Guide(dim=256)
        self.resnet = models.resnet18(pretrained=False)
        # # self.basemodel = backbone
        self.basemodel = Net(self.resnet)
        # self.basemodel = My_Res()
        # self.basemodel = Blocks()
        # self.vgg = models.vgg19_bn(pretrained=True)
        # self.basemodel = backbone
        # self.basemodel = Net_vgg(self.vgg)

        self.trt_fea_model = None
        self.trt_track_model = None
        self._phase = "train"
        # self.vit = ViT(image_size=224,patch_size = 32,
        #       num_classes = 1000,
        #       dim = 1024,
        #       depth = 6,
        #       heads = 16,
        #       mlp_dim = 2048,
        #       dropout = 0.1,
        #       emb_dropout = 0.1)

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        assert p in self.support_phases
        self._phase = p

    def forward_transformer(self,f_z,e_z):
        pass

    def train_forward(self, training_data):
        target_img = training_data["em_z"].unsqueeze(1)
        search_img = training_data["em_x"].unsqueeze(1)
        # backbone feature
        # target_img = tvisf.normalize(target_img, self.mean, self.std)
        # search_img = tvisf.normalize(search_img, self.mean, self.std)
        f_z = self.basemodel(target_img)
        # ff_z = self.basemodel(target_frame)
        f_x = self.basemodel(search_img)

        # f_z = self.vit(f_z)
        # fused_z = self.vit(f_z,e_z)
        # feature adjustment
        # fuse_fe = self.attention(f_z,ff_z)

        c_z_k = self.c_z_k(f_z)
        # r_z_k = self.r_z_k(f_z)
        c_x = self.c_x(f_x)
        # r_x = self.r_x(f_x)
        # feature matching
        # r_out = xcorr_depthwise(r_x, r_z_k)
        c_out = xcorr_depthwise(c_x, c_z_k)
        # fused_model = self.attention(r_out,c_out) # B X 256 X 28 X 28
        # head
        out,outputs_coord = self.head(
            c_out,c_x)
        predict_data = dict(
            # cls_pred=fcos_cls_score_final,
            # ctr_pred=fcos_ctr_score_final,
            box_pred=out['pred_boxes'],
        )
        # if self._hyper_params["corr_fea_output"]:
        #     predict_data["corr_fea"] = corr_fea
        return predict_data

    def instance(self, img):
        f_z = self.basemodel(img)
        # template as kernel
        c_x = self.c_x(f_z)
        self.cf = c_x

    def forward(self, *args, phase=None):
        r"""
        Perform tracking process for different phases (e.g. train / init / track)

        Arguments
        ---------
        target_img: torch.Tensor
            target template image patch
        search_img: torch.Tensor
            search region image patch

        Returns
        -------
        fcos_score_final: torch.Tensor
            predicted score for bboxes, shape=(B, HW, 1)
        fcos_bbox_final: torch.Tensor
            predicted bbox in the crop, shape=(B, HW, 4)
        fcos_cls_prob_final: torch.Tensor
            classification score, shape=(B, HW, 1)
        fcos_ctr_prob_final: torch.Tensor
            center-ness score, shape=(B, HW, 1)
        """
        if phase is None:
            phase = self._phase
        # used during training
        if phase == 'train':
            # resolve training data
            if self._hyper_params["amp"]:
                with torch.cuda.amp.autocast():
                    return self.train_forward(args[0])
            else:
                return self.train_forward(args[0])

        # used for template feature extraction (normal mode)
        elif phase == 'feature':
            target_event, = args
            if self._hyper_params["trt_mode"]:
                # extract feature with trt model
                out_list = self.trt_fea_model(target_event)
            else:
                # backbone feature
                f_z = self.basemodel(target_event)
                # ff_z = self.basemodel(target_img)

                # template as kernel
                # f_z = self.vit(f_z)
                # fuse_fe = self.attention(f_z, ff_z)

                c_z_k = self.c_z_k(f_z)
                # r_z_k = self.r_z_k(f_z)

                # output
                out_list = [c_z_k, c_z_k]
        # used for template feature extraction (trt mode)
        elif phase == "freeze_track_fea":
            search_img, = args
            # backbone feature
            f_x = self.basemodel(search_img)
            # feature adjustment
            c_x = self.c_x(f_x)
            r_x = self.r_x(f_x)
            # head
            return [c_x, r_x]
        # [Broken] used for template feature extraction (trt mode)
        #   currently broken due to following issue of "torch2trt" package
        #   c.f. https://github.com/NVIDIA-AI-IOT/torch2trt/issues/251
        elif phase == "freeze_track_head":
            c_out, r_out = args
            # head
            outputs = self.head(c_out, r_out, 0, True)
            return outputs
        # used for tracking one frame during test
        elif phase == 'track':
            if len(args) == 3:
                search_event, c_z_k, r_z_k = args
                if self._hyper_params["trt_mode"]:
                    c_x, r_x = self.trt_track_model(search_event)
                else:
                    # backbone feature
                    # time_start1 = time.time()
                    f_x = self.basemodel(search_event)
                    # feature adjustment
                    # time_start2 = time.time()
                    # print('backbone',time_start2-time_start1)
                    c_x = self.c_x(f_x)
                    # r_x = self.r_x(f_x)
            elif len(args) == 4:
                # c_x, r_x already computed
                c_z_k, r_z_k, c_x, r_x = args
            else:
                raise ValueError("Illegal args length: %d" % len(args))

            # feature matching
            # r_out = xcorr_depthwise(r_x, r_z_k)
            c_out = xcorr_depthwise(c_x, c_z_k)
            # time_start1 = time.time()
            # feature adjustment

            out, outputs_coord = self.head(
                c_out,c_x)
            # time_start2 = time.time()
            # print('transformer', time_start2 - time_start1)
            # extra = dict(c_x=c_x)

            # if self._hyper_params["corr_fea_output"]:
            #     predict_data["corr_fea"] = corr_fea

            out_list = out['pred_boxes'], out
        else:
            raise ValueError("Phase non-implemented.")

        return out_list

    def update_params(self):
        r"""
        Load model parameters
        """
        self._make_convs()
        self._initialize_conv()
        super().update_params()
        if self._hyper_params["trt_mode"]:
            logger.info("trt mode enable")
            from torch2trt import TRTModule
            self.trt_fea_model = TRTModule()
            self.trt_fea_model.load_state_dict(
                torch.load(self._hyper_params["trt_fea_model_path"]))
            self.trt_track_model = TRTModule()
            self.trt_track_model.load_state_dict(
                torch.load(self._hyper_params["trt_track_model_path"]))
            logger.info("loading trt model succefully")

    def _make_convs(self):
        head_width = self._hyper_params['head_width']

        # feature adjustment
        # self.r_z_k = conv_bn_relu(head_width,
        #                           head_width,
        #                           1,
        #                           3,
        #                           0,
        #                           has_relu=False)
        self.c_z_k = conv_bn_relu(head_width,
                                  head_width,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        # self.r_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [
            self.c_z_k.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
