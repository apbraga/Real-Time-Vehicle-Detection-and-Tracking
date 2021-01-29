import dependencies
import glob
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import argparse
from utils import flow_viz
#---------------------------------------------------------------
#GLOBAL VARIABLES-----------------------------------------------
#---------------------------------------------------------------
ID = 0
FRAME_NUMBER = 1
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
DEVICE = 'cuda'
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
class InputData:
    def __init__(self, media_type, location):
        self.type = media_type
        if self.type == 'image':
            self.images = glob.glob(os.path.join(location))
            self.images = sorted(self.images)

        if self.type == 'video':
            self.images = cv2.VideoCapture(location)

    def get_next_frame(self):
        try:
            if self.type == 'image':
                imfile = self.images[0]
                image = np.array(Image.open(imfile)).astype(np.uint8)
                image = cv2.resize(image, (320,240), interpolation = cv2.INTER_AREA)
                return image

            if self.type == 'video':
                _, image = self.images.read()
                image = cv2.resize(image, (320,240), interpolation = cv2.INTER_AREA)
                return image
        except:
            return False

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
class Vehicle:
    def __init__(self,frame_number, detection, flow):
        global ID
        global FRAME_NUMBER
        self.x, self.y, self.w, self.h = detection
        self.x_dot, self.y_dot = flow
        self.first_frame = FRAME_NUMBER
        self.last_seen = FRAME_NUMBER
        self.veh_id = ID
        ID = ID + 1

    def update_full(self, frame_number, detection, flow):
        self.last_seen = FRAME_NUMBER

    def update_partial(self, flow):
        self.x, self.y
        self.x_dot, self.y_dot

    def predict(self):
        self.x = self.x + self.x_dot
        self.y = self.y + self.y_dot

    def bounds(self):
        return True
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
class Frame:
    def __init__(self, detection, flow , vehicles):
        global FRAME_NUMBER
        self.frame_number = FRAME_NUMBER
        self.bounding_boxes = detection
        self.optical_flow = flow
        self.prior_vehicles = vehicles
        self.measurement = {}
        self.update_veh = []
        self.predict_veh = []
        FRAME_NUMBER = FRAME_NUMBER + 1

    def match(self):
        error = np.zeros((len(self.bounding_boxes),len(self.prior_vehicles)))
        for i in range(len(self.bounding_boxes)):
            for j in range(len(self.prior_vehicles)):
                error[i][j] = np.sqrt((self.bounding_boxes[i][0]-self.prior_vehicles[j].x)**2+(self.bounding_boxes[i][1]-self.prior_vehicles[j].y)**2)

        for i in range(len(self.bounding_boxes)):
            if min(error[i]) < 1.0:
                idx = int(np.where(error[i] == error[i].min())[0])
                self.measurement[self.prior_vehicles[idx].veh_id] = ['full',self.bounding_boxes[i], self.average_flow(self.bounding_boxes[i]), self.prior_vehicles[idx]]
            else:
                vehicle = Vehicle(self.frame_number, self.bounding_boxes[i], self.average_flow(self.bounding_boxes[i]))
                self.measurement[vehicle.veh_id] = ['initialize', 0, 0, vehicle]

        for vehicle in self.prior_vehicles:
            if vehicle.last_seen - self.frame_number > 10:
                if vehicle.veh_id not in self.measurement:
                    bbox = [vehicle.x, vehicle.y, vehicle.w, vehicle.h]
                    self.measurement[vehicle.veh_id] = ['partial', 0, self.average_flow(bbox), vehicle]

    def update(self):
        for item in self.measurement.values():
            if item[0] == 'full':
                item[3].update_full(self.frame_number, item[1], item[2])
            if item[0] == 'partial':
                item[3].update_partial(item[2])
            self.update_veh.append(item[3])

    def predict(self):
        for vehicle in self.update_veh:
            vehicle.predict()
            if vehicle.bounds():
                self.predict_veh.append(vehicle)

    def average_flow(self, bbox):
        x, y, w, h = bbox
        magnitude = np.average(self.optical_flow[0][0][x:x+w , y : y +h])
        direction = np.average(self.optical_flow[0][1][x:x+w , y : y +h])
        return [magnitude, direction]

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
class Detector:
    def __init__(self, type):
        self.type = type
        if type == 'frcnn':
            # load faster r-cnn
            self.detector = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            # send model to gpu
            self.detector.to(DEVICE)
            # set model to inference mode
            self.detector.eval()
            # set transformation to prepare image for network input
            self.transform = transforms.Compose([transforms.ToTensor()])
        if type == 'yolo':
            return True
        if type == 'sinet':
            return True

    def inference(self, image):
        if self.type == 'frcnn':
            # convert image to torch tensor
            image = self.transform(image)
            # send input data to GPU
            image = image.to(DEVICE)
            # process inference and get detections
            detections = self.detector([image])
            boxes = detections[0]['boxes']
            confidence = detections[0]['scores']
            class_id = detections[0]['labels']



        if self.type == 'yolo':
            detections, confidence, class_id = self.detector([image])
            detections.detach().cpu().numpy()
            confidence.detach().cpu().numpy()
            class_id.detach().cpu().numpy()

        if self.type == 'sinet':
            detections, confidence, class_id = self.detector([image])
            detections.detach().cpu().numpy()
            confidence.detach().cpu().numpy()
            class_id.detach().cpu().numpy()


        self.result = self.filter_detection(boxes, confidence, class_id)

    def filter_detection(self, detections, confidence, class_id):
        x1 = detections[:, 0].detach().cpu().numpy()
        y1 = detections[:, 1].detach().cpu().numpy()
        x2 = detections[:, 2].detach().cpu().numpy()
        y2 = detections[:, 3].detach().cpu().numpy()
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

            inds = np.where(ovr <= IOU_THRESHOLD)[0]
            order = order[inds + 1]
        filter = []
        for i in keep:
            if confidence[i] > SCORE_THRESHOLD:
                if class_id[i] in [2,3,4,6,8]:
                    filter.append([int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])])
        return filter

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
class OpticalFlow:
    def __init__(self, type):
        self.type = type
        if type == 'farneback':
            return True

        if type == 'raft':
            parser = argparse.ArgumentParser()
            parser.add_argument('--model', nargs='?', const='raft-models/raft-things.pth', type=str, help="restore checkpoint")
            parser.add_argument('--path', nargs='?', const='frames', type=int, help="dataset for evaluation")
            parser.add_argument('--small', action='store_true', help='use small model')
            parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
            parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
            args = parser.parse_args()
            self.flow_model = torch.nn.DataParallel(RAFT(args))
            self.flow_model.load_state_dict(torch.load(args.model))
            self.flow_model = self.flow_model.module
            self.flow_model.to(DEVICE)
            self.flow_model.eval()

        if type == 'flownet':
            return True


    def inference(self,image1,image2):
        if self.type == 'farneback':
            self.flow = True
        if self.type == 'raft':
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
            image1 = image1[None].to(DEVICE)

            image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
            image2 = image2[None].to(DEVICE)

            padder = InputPadder(image2.shape)
            image1, image2 = padder.pad(image1, image2)
            _, flow_up = self.flow_model(image1, image2, iters=5, test_mode=True)
            self.result = flow_up.detach().cpu().numpy()

        if self.type == 'flownet':
            self.flow = True



    def toimage(self):
        image = flow_viz.flow_to_image(self.flow)
        return image
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------


def draw_bbox(image, bboxes):
    for bbox in bboxes:
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), np.random.uniform(0, 255) , 2)
    return image

def flow_mask(flow, bboxes):
    image = flow.toimage()
    mask = np.full(image.shape[:2], 0, dtype=np.uint8)
    for bbox in bboxes:
        cv2.rectangle(mask, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), -1)
    image = cv2.bitwise_or(np.array(image).astype(np.uint8), np.array(image).astype(np.uint8), mask=mask)
    return image

def visuzalization(input, bbox, flow):
    bbox = draw_bbox(input, bbox)
    flow_visualization = flow.toimage()
    mask = flow_mask(flow, bbox)

    img1 = np.concatenate((input,bbox), axis = 1)
    img2 = np.concatenate((flow_visualization, mask), axis = 1)
    merge_img = np.concatenate((img1,img2), axis = 1)
    return merge_img

def display_result(image):
    cv2.imshow('image',image)
    cv2.waitKey(1)
