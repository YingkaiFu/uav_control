# -*- coding: utf-8 -*
import time
from typing import List

# from PIL import Image
import cv2
import numpy as np
import torch
from evaluation.got_benchmark.utils.viz import show_frame
from pipeline.pipeline_base import PipelineBase

def load_pth(img_file: str) -> np.array:
    start_event = torch.load(img_file)
    numpy_image = (start_event).numpy().transpose(1,2,0)
    # numpy_image[numpy_image>0]=255
    # numpy_image[numpy_image==0]=127
    # numpy_image[numpy_image<0]=0
    # numpy_image = (numpy_image-np.min(numpy_image))/(np.max(numpy_image)-np.min(numpy_image))*255
    # output = np.concatenate((numpy_image, numpy_image, numpy_image), axis=-1)
    return numpy_image

class PipelineTracker(object):
    def __init__(self,
                 name: str,
                 pipeline: PipelineBase,
                 is_deterministic: bool = True):
        """Helper tracker for comptability with 
        
        Parameters
        ----------
        name : str
            [description]
        pipeline : PipelineBase
            [description]
        is_deterministic : bool, optional
            [description], by default False
        """
        self.name = name
        self.is_deterministic = is_deterministic
        self.pipeline = pipeline

    def init(self,event: np.array, box):
        """Initialize pipeline tracker
        
        Parameters
        ----------
        image : np.array
            image of the first frame
        box : np.array or List
            tracking bbox on the first frame
            formate: (x, y, w, h)
        """
        self.pipeline.init(event, box)

    def update(self, event: np.array):
        """Perform tracking
        
        Parameters
        ----------
        image : np.array
            image of the current frame
        
        Returns
        -------
        np.array
            tracking bbox
            formate: (x, y, w, h)
        """
        return self.pipeline.update(event)

    def track(self, events: List, box, visualize: bool = False):
        """Perform tracking on a given video sequence
        
        Parameters
        ----------
        img_files : List
            list of image file paths of the sequence
        box : np.array or List
            box of the first frame
        visualize : bool, optional
            Visualize or not on each frame, by default False
        
        Returns
        -------
        [type]
            [description]
        """
        fix_num = 1957
        frame_num = len(events)
        boxes = np.zeros((frame_num, 4))
        times = np.zeros(frame_num)
        flag = True
        for f in range(frame_num):
            # image = Image.open(img_file)
            # if not image.mode == 'RGB':
            #     image = image.convert('RGB')\
            event = load_pth(events[f])
            start_time = time.time()
            if (box[f] == 0).all() == False and flag:
                boxes[f] = box[f]
                self.init(event, box[f])
                flag = False
            elif not flag:
                if f==fix_num:
                    vv = 10
                boxes[f, :],extra = self.update(event)
                # self.init(image, event, boxes[f, :])
            times[f] = time.time() - start_time

            if visualize:
                show_frame(boxes[f, :])

        return boxes, times

