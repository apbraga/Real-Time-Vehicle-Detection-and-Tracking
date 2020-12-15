import glob
import time
import cv2 as cv
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
# Define GPU usage
DEVICE = 'cuda'
# Set input video
cap = cv.VideoCapture("1-fps.mp4")
# get and drop first Frame
ret, first_frame = cap.read()
# load faster r-cnn to gpu
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(DEVICE)
# set model to inference mode
model.eval()
# set transformation to prepare image for network input
transform = transforms.Compose([transforms.ToTensor()])
# Run until video is finished
while(cap.isOpened()):
    start = time.time()
    # get frame
    ret, frame = cap.read()
    # convert image to torch tensor
    img = transform(frame)
    # send input data to GPU
    img = img.to(DEVICE)
    # process inference
    predictions = model([img])
    print('elapsed time: {}'.format(time.time()-start))
