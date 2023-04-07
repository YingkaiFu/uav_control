# -*- coding: utf-8 -*

from copy import deepcopy

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pipeline.pipeline_base import TRACK_PIPELINES, PipelineBase
from pipeline.utils import (cxywh2xywh, get_crop,
                                         get_subwindow_tracking,
                                         imarray_to_tensor, tensor_to_numpy,
                                         xywh2cxywh, xyxy2cxywh)


# ============================== Tracker definition ============================== #
@TRACK_PIPELINES.register
class SiamFCppTracker(PipelineBase):
    r"""
    Basic SiamFC++ tracker

    Hyper-parameters
    ----------------
        total_stride: int
            stride in backbone
        context_amount: float
            factor controlling the image patch cropping range. Set to 0.5 by convention.
        test_lr: float
            factor controlling target size updating speed
        penalty_k: float
            factor controlling the penalization on target size (scale/ratio) change
        window_influence: float
            factor controlling spatial windowing on scores
        windowing: str
            windowing type. Currently support: "cosine"
        z_size: int
            template image size
        x_size: int
            search image size
        num_conv3x3: int
            number of conv3x3 tiled in head
        min_w: float
            minimum width
        min_h: float
            minimum height
        phase_init: str
            phase name for template feature extraction
        phase_track: str
            phase name for target search
        corr_fea_output: bool
            whether output corr feature

    Hyper-parameters (to be calculated at runtime)
    ----------------------------------------------
    score_size: int
        final feature map
    score_offset: int
        final feature map
    """
    default_hyper_params = dict(
        total_stride=8,
        score_size=17,
        score_offset=87,
        context_amount=0.5,
        test_lr=0.52,
        penalty_k=0.04,
        window_influence=0.21,
        windowing="cosine",
        z_size=127,
        x_size=303,
        num_conv3x3=3,
        min_w=10,
        min_h=10,
        phase_init="feature",
        phase_track="track",
        corr_fea_output=False,
    )

    def __init__(self, *args, **kwargs):
        super(SiamFCppTracker, self).__init__(*args, **kwargs)
        self.update_params()

        # set underlying model to device
        self.device = torch.device("cpu")
        self.debug = False
        self.set_model(self._model)

    def set_model(self, model):
        """model to be set to pipeline. change device & turn it into eval mode

        Parameters
        ----------
        model : ModuleBase
            model to be set to pipeline
        """
        self._model = model.to(self.device)
        self._model.eval()

    def set_device(self, device):
        self.device = device
        self._model = self._model.to(device)

    def update_params(self):
        hps = self._hyper_params
        hps['score_size'] = (
                                    hps['x_size'] -
                                    hps['z_size']) // hps['total_stride'] + 1 - hps['num_conv3x3'] * 2
        hps['score_offset'] = (
                                      hps['x_size'] - 1 -
                                      (hps['score_size'] - 1) * hps['total_stride']) // 2
        self._hyper_params = hps

    def feature(self,em, target_pos, target_sz, avg_chans2=None):
        """Extract feature

        Parameters
        ----------
        im : np.array
            initial frame
        target_pos :
            target position (x, y)
        target_sz : [type]
            target size (w, h)
        avg_chans : [type], optional
            channel mean values, (B, G, R), by default None

        Returns
        -------
        [type]
            [description]
        """
        if avg_chans2 is None:
            avg_chans2 = np.mean(em, axis=(0, 1))
        z_size = self._hyper_params['z_size']
        context_amount = self._hyper_params['context_amount']

        em_z_crop, _ = get_crop(
            em,
            target_pos,
            target_sz,
            z_size,
            avg_chans=avg_chans2,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )

        phase = self._hyper_params['phase_init']
        with torch.no_grad():
            data2 = imarray_to_tensor((np.expand_dims(em_z_crop,axis=-1))).to(self.device)
            features = self._model(data2, phase=phase)

        return features, em_z_crop, avg_chans2

    def init(self,em, state):
        r"""Initialize tracker
            Internal target state representation: self._state['state'] = (target_pos, target_sz)

        Arguments
        ---------
        im : np.array
            initial frame image
        state
            target state on initial frame (bbox in case of SOT), format: xywh
        """
        rect = state  # bbox in xywh format is given for initialization in case of tracking
        box = xywh2cxywh(rect)
        target_pos, target_sz = box[:2], box[2:]

        self._state['im_h'] = em.shape[0]
        self._state['im_w'] = em.shape[1]

        # extract template feature
        features, em_z_crop, avg_chans_e = self.feature(em, target_pos, target_sz)

        score_size = self._hyper_params['score_size']

        self._state['features'] = features
        self._state['z_crop_e'] = em_z_crop
        self._state['avg_chans_e'] = avg_chans_e
        # self.state['target_pos'] = target_pos
        # self.state['target_sz'] = target_sz
        self._state['state'] = (target_pos, target_sz)

    def get_avg_chans(self):
        return self._state['avg_chans']

    def track(self,
              em_x,
              target_pos,
              target_sz,
              features,
              update_state=False,
              **kwargs):

        if 'avg_chans_e' in kwargs:
            avg_chans_e = kwargs['avg_chans_e']
        else:
            avg_chans_e = self._state['avg_chans_e']

        z_size = self._hyper_params['z_size']
        x_size = self._hyper_params['x_size']
        context_amount = self._hyper_params['context_amount']
        phase_track = self._hyper_params['phase_track']
        em_x_crop, scale_x = get_crop(
            em_x,
            target_pos,
            target_sz,
            z_size,
            x_size=x_size,
            avg_chans=avg_chans_e,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )

        self._state["scale_x"] = deepcopy(scale_x)
        with torch.no_grad():
            box, extra = self._model(
                imarray_to_tensor(np.expand_dims(em_x_crop,axis=-1)).to(self.device),
                *features,
                phase=phase_track)
        if self._hyper_params["corr_fea_output"]:
            self._state["corr_fea"] = extra["corr_fea"]

        box = tensor_to_numpy(box[0][0]) * x_size
        x_c, y_c, w, h = box

        # score post-processing
        box = box/ np.float32(scale_x)
        res_x = target_pos[0] + box[0]-(x_size//2)/scale_x
        res_y = target_pos[1] + box[1]-(x_size//2)/scale_x

        new_target_pos = np.array([res_x,res_y])
        new_target_sz = np.array([box[2],box[3]])
        # restrict new_target_pos & new_target_sz
        new_target_pos, new_target_sz = self._restrict_box(
            new_target_pos, new_target_sz)
        # crop = deepcopy(im_x)
        # a1,a2 = int(new_target_pos[0]-0.5*new_target_sz[0]),int(new_target_pos[1]-0.5*new_target_sz[1])
        # cv2.rectangle(crop, (a1, a2),
        #               (int(a1 + new_target_sz[0]), int(a2 + new_target_sz[1])),
        #               (255, 0, 255), 1)
        # plt.imshow(crop)
        # plt.show()

        # print((int(new_target_pos[0]), int(new_target_pos[1])),
        #       (int(new_target_pos[0] + new_target_sz[0]), int(new_target_pos[1] + new_target_sz[1])))
        # cv2.waitKey(0)
        # record basic mid-level info
        self._state['x_crop'] = em_x_crop
        # record optional mid-level info
        if update_state:
            self._state['all_box'] = box

        return new_target_pos, new_target_sz, extra

    def set_state(self, state):
        self._state["state"] = state

    def get_track_score(self):
        return float(self._state["pscore"])

    def update(self,em, state=None):
        """ Perform tracking on current frame
            Accept provided target state prior on current frame
            e.g. search the target in another video sequence simutanously

        Arguments
        ---------
        im : np.array
            current frame image
        state
            provided target state prior (bbox in case of SOT), format: xywh
        """
        # use prediction on the last frame as target state prior
        if state is None:
            target_pos_prior, target_sz_prior = self._state['state']
        # use provided bbox as target state prior
        else:
            rect = state  # bbox in xywh format is given for initialization in case of tracking
            box = xywh2cxywh(rect).reshape(4)
            target_pos_prior, target_sz_prior = box[:2], box[2:]
        features = self._state['features']

        # forward inference to estimate new state
        target_pos, target_sz, extra = self.track(em,
                                           target_pos_prior,
                                           target_sz_prior,
                                           features,
                                           update_state=True)

        # save underlying state
        # self.state['target_pos'], self.state['target_sz'] = target_pos, target_sz
        self._state['state'] = target_pos, target_sz

        # return rect format
        track_rect = cxywh2xywh(np.concatenate([target_pos, target_sz],
                                               axis=-1))
        if self._hyper_params["corr_fea_output"]:
            return target_pos, target_sz, self._state["corr_fea"]
        return track_rect, extra

    # ======== tracking processes ======== #

    def _postprocess_score(self, score, box_wh, target_sz, scale_x):
        r"""
        Perform SiameseRPN-based tracker's post-processing of score
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_sz: previous state (w & h)
        :param scale_x:
        :return:
            best_pscore_id: index of chosen candidate along axis HW
            pscore: (HW, ), penalized score
            penalty: (HW, ), penalty due to scale/ratio change
        """

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        penalty_k = self._hyper_params['penalty_k']
        target_sz_in_crop = target_sz * scale_x
        s_c = change(
            sz(box_wh[:, 2], box_wh[:, 3]) /
            (sz_wh(target_sz_in_crop)))  # scale penalty
        r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) /
                     (box_wh[:, 2] / box_wh[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
        pscore = penalty * score

        # ipdb.set_trace()
        # cos window (motion model)
        window_influence = self._hyper_params['window_influence']
        pscore = pscore * (
                1 - window_influence) + self._state['window'] * window_influence
        best_pscore_id = np.argmax(pscore)

        return best_pscore_id, pscore, penalty

    def _postprocess_box(self, best_pscore_id, score, box_wh, target_pos,
                         target_sz, scale_x, x_size, penalty):
        r"""
        Perform SiameseRPN-based tracker's post-processing of box
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_pos: (2, ) previous position (x & y)
        :param target_sz: (2, ) previous state (w & h)
        :param scale_x: scale of cropped patch of current frame
        :param x_size: size of cropped patch
        :param penalty: scale/ratio change penalty calculated during score post-processing
        :return:
            new_target_pos: (2, ), new target position
            new_target_sz: (2, ), new target size
        """
        pred_in_crop = box_wh[best_pscore_id, :] / np.float32(scale_x)
        # about np.float32(scale_x)
        # attention!, this casting is done implicitly
        # which can influence final EAO heavily given a model & a set of hyper-parameters

        # box post-postprocessing
        test_lr = self._hyper_params['test_lr']
        lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr
        res_x = pred_in_crop[0] + target_pos[0] - (x_size // 2) / scale_x
        res_y = pred_in_crop[1] + target_pos[1] - (x_size // 2) / scale_x
        res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
        res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

        new_target_pos = np.array([res_x, res_y])
        new_target_sz = np.array([res_w, res_h])

        return new_target_pos, new_target_sz

    def _restrict_box(self, target_pos, target_sz):
        r"""
        Restrict target position & size
        :param target_pos: (2, ), target position
        :param target_sz: (2, ), target size
        :return:
            target_pos, target_sz
        """
        target_pos[0] = max(0, min(self._state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(self._state['im_h'], target_pos[1]))
        target_sz[0] = max(self._hyper_params['min_w'],
                           min(self._state['im_w'], target_sz[0]))
        target_sz[1] = max(self._hyper_params['min_h'],
                           min(self._state['im_h'], target_sz[1]))

        return target_pos, target_sz

    def _cvt_box_crop2frame(self, box_in_crop, target_pos, scale_x, x_size):
        r"""
        Convert box from cropped patch to original frame
        :param box_in_crop: (4, ), cxywh, box in cropped patch
        :param target_pos: target position
        :param scale_x: scale of cropped patch
        :param x_size: size of cropped patch
        :return:
            box_in_frame: (4, ), cxywh, box in original frame
        """
        x = (box_in_crop[..., 0]) / scale_x + target_pos[0] - (x_size //
                                                               2) / scale_x
        y = (box_in_crop[..., 1]) / scale_x + target_pos[1] - (x_size //
                                                               2) / scale_x
        w = box_in_crop[..., 2] / scale_x
        h = box_in_crop[..., 3] / scale_x
        box_in_frame = np.stack([x, y, w, h], axis=-1)

        return box_in_frame
