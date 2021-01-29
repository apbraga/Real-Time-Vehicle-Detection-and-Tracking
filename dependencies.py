import sys
sys.path.append('core')
import os
import argparse
import glob
import time
import cv2 as cv
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from PIL import Image

# detector class names
coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

ID = 0
FRAME_NUMBER = 0

# utillity fuction for no-max-suppression, that filters ambiguous bounding boxes
def nms(dets, confidence, thresh):
    x1 = dets[:, 0].detach().cpu().numpy()
    y1 = dets[:, 1].detach().cpu().numpy()
    x2 = dets[:, 2].detach().cpu().numpy()
    y2 = dets[:, 3].detach().cpu().numpy()
    scores = confidence.detach().cpu().numpy()
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        j = order[0]
        keep.append(j)
        xx1 = np.maximum(x1[j], x1[order[1:]])
        yy1 = np.maximum(y1[j], y1[order[1:]])
        xx2 = np.minimum(x2[j], x2[order[1:]])
        yy2 = np.minimum(y2[j], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[j] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# Calculate model metrics
def evaluate(ground_truth, inference):
    return true_positive, true_negative, false_positive, false_negative
