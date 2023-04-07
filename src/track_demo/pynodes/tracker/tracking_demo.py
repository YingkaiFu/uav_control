# -*- coding: utf-8 -*
import time
import sys
path = sys.path[0]
path = path + '/../../src/pylibs'
sys.path.append(path)
from config.config import cfg, specify_task
from model import builder as model_builder
from pipeline import builder as pipeline_builder
import numpy as np
from pathlib import Path
import torch
import tqdm
import cv2
from loguru import logger


def load_pth(img_file: str) -> np.array:
    start_event = torch.load(img_file)
    numpy_image = (start_event).numpy().transpose(1,2,0)
    # numpy_image = (numpy_image-np.min(numpy_image))/(np.max(numpy_image)-np.min(numpy_image))*255
    # output = np.concatenate((numpy_image, numpy_image, numpy_image), axis=-1)
    return numpy_image



conf = "src/track_demo/config/tracker/img_ext_dataset.yaml"
root_cfg = cfg
root_cfg.merge_from_file(conf)
logger.info("Load experiment configuration at: %s" % conf)

# resolve config
root_cfg = root_cfg.test
task, task_cfg = specify_task(root_cfg)
model = model_builder.build(task, task_cfg.model)

pipeline = pipeline_builder.build(task, task_cfg.pipeline, model)
dev = torch.device("cuda:0")
pipeline.set_device(dev)
# init_box = [100,100,100,100]
template = None
# loop over sequence
# dataset_root = '/home/yingkai/dataset/checkerboard/240_event'
# gt_file = '/home/yingkai/dataset/checkerboard/groundtruth.txt'
dataset_root = '/home/yingkai/dataset/uav1/240_event'
gt_file = '/home/yingkai/dataset/uav/groundtruth.txt'
# dataset_root = '/home/yingkai/event_camera/code/dataset/fe_data/val/truck222_4/240_event'
# gt_file = '/home/yingkai/event_camera/code/dataset/fe_data/val/truck222_4/groundtruth.txt'
show = True


# anno = np.loadtxt(gt_file, delimiter=',')
out_bbox = []

file_list = sorted(Path(dataset_root).iterdir())
time1 = time.time()
select = False
skip=410+87
for index,file in tqdm.tqdm(enumerate(file_list[skip:])):
    if select == False:
        frame = load_pth(file)
        box = cv2.selectROI("output",
                            frame,
                            fromCenter=False,
                            showCrosshair=True)
        out_bbox.append(box)
        select =True
        pipeline.init(frame,box)
    else:
        frame = load_pth(file)
        # time1 = time.time()
        rect_pred = pipeline.update(frame)
        out_bbox.append(rect_pred[0])
    if show:
        output = np.concatenate((frame, frame, frame), axis=-1)
        cv2.rectangle(output, (int(out_bbox[index][0]), int(out_bbox[index][1])), (int(out_bbox[index][0]) + int(out_bbox[index][2]), int(out_bbox[index][1]) + int(out_bbox[index][3])),
                [1,0,0], 2)
        # cv2.rectangle(output, (int(anno[index][0]), int(anno[index][1])), (int(anno[index][0]) + int(anno[index][2]), int(anno[index][1]) + int(anno[index][3])),
        #         [0,1,0], 2)
        cv2.imshow('output',output)
        key = cv2.waitKey(0)
        if key==ord('q'):
            break

print(time.time()-time1)
        # print(time.time()-time1)
np.savetxt('predict.txt',out_bbox,delimiter=',')