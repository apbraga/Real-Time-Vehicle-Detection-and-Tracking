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

# tresholding for bouding box selection
SCORE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.5

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
# input form
# python frcnn.py --model=models/raft-things.pth --path=demo-frames
parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--path', help="dataset for evaluation")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()
# Define GPU usage
DEVICE = 'cuda'
# Set input video
cap = cv.VideoCapture("1-fps.mp4")
# get and drop first Frame
ret, current_img = cap.read()
count = 1
###############################################################################
# FASTER REGION CONVOLUTIONAL NETWORK
# load faster r-cnn to gpu
frcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
frcnn.to(DEVICE)
# set model to inference mode
frcnn.eval()
# set transformation to prepare image for network input
transform = transforms.Compose([transforms.ToTensor()])
###############################################################################
# Recurrent All-Pairs Field Transforms for Optical Flow
raft = torch.nn.DataParallel(RAFT(args))
raft.load_state_dict(torch.load(args.model))
raft = raft.module
raft.to(DEVICE)
raft.eval()
###############################################################################
#ouput video setup
#h, w = current_img.shape[:2]
#fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#out = cv2.VideoWriter("output.mp4", fourcc, 5.0, (2*w, 2*h))
###############################################################################
# Run until video is finished
while(cap.isOpened()):
    count = count +1
    start = time.time()
    # get frame
    ret, frame = cap.read()
    frame_1 = frame.copy()
    # convert image to torch tensor
    frcnn_img = transform(frame)
    # send input data to GPU
    frcnn_img = frcnn_img.to(DEVICE)
    # process inference
    detections = frcnn([frcnn_img])

    boxes = detections[0]['boxes']
    confidences = detections[0]['scores']
    class_id = detections[0]['labels']

    bbox = frame

    idxs = nms(boxes,confidences, IOU_THRESHOLD)

    for i in idxs:
        if confidences[i] > SCORE_THRESHOLD:
            if class_id[i] in [2,3,4,6,8]:
                color = COLORS[5]
                cv2.rectangle(bbox, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), color, 2)
        #cv2.imwrite('detection{:06d}'.format(count) +  '.png',image)

    ############################################################################
    past_img = current_img
    current_img = frame_1

    raft_img_1 = np.array(cv2.medianBlur(past_img,5)).astype(np.uint8)
    raft_img_1 = torch.from_numpy(raft_img_1).permute(2, 0, 1).float()
    raft_img_1 = raft_img_1[None].to(DEVICE)

    raft_img_2 = np.array(cv2.medianBlur(current_img,5)).astype(np.uint8)
    raft_img_2 = torch.from_numpy(raft_img_2).permute(2, 0, 1).float()
    raft_img_2 = raft_img_2[None].to(DEVICE)

    #padder = InputPadder(raft_img_1.shape)
    #raft_img_2, raft_img_1 = padder.pad(raft_img_2, raft_img_1)
    flow_low, flow_up = raft(raft_img_2, raft_img_1, iters=5, test_mode=True)

    flow_up = flow_up[0].permute(1,2,0).detach().cpu().numpy()
    flow_up = flow_viz.flow_to_image(flow_up)
    #cv2.imwrite('{:06d}'.format(count) +  '.png',flow_up)

    merge = cv2.addWeighted(bbox, 1, flow_up, .5, 0)
    merge_img1 = np.concatenate((frame_1,bbox), axis = 0)
    merge_img2 = np.concatenate((flow_up, merge), axis = 0)
    merge_img = np.concatenate((merge_img1,merge_img2), axis = 1)
    cv2.imshow('image',merge_img)
    cv2.waitKey(1)
    #out.write(merge_img)
    #cv2.imwrite('{:06d}'.format(count) +  '.png',merge_img)
    print('elapsed time: {}'.format(time.time()-start))

cap.release()
cv2.destroyAllWindows()
