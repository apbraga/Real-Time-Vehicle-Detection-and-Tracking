import sys
import dependencies
sys.path.append('core')
import math
import glob
import os
import time
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
#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------
# Set Input type - image, video
media_type = 'image'
# Location of image folder or video file
location = 'dataset/01/'
# Set detector model type - frcnn, yolo, sinet
detector_type = 'frcnn'
# Set optical flow model type - farneback, raft, flownet
flow_type = 'raft'
# Set output types - metrics, video, preview

results_out = False
time_out = True
video_out = False
preview_result = False
metrics_out = True
performance = [0,0,0]
#---------------------------------------------------------------
# Global Variables
#---------------------------------------------------------------
ID = 1
FRAME_NUMBER = 2
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
DEVICE = 'cuda'
IMG_X_MAX = 320
IMG_Y_MAX = 240

#---------------------------------------------------------------
# Input Image Sequence Handler Class
#---------------------------------------------------------------
class InputData:
    # Initializer handling both image and video as input data
    def __init__(self, media_type, location):
        self.type = media_type
        if self.type == 'image':
            self.images = glob.glob(os.path.join(location, '*.png'))
            self.images = sorted(self.images)

        if self.type == 'video':
            self.images = cv2.VideoCapture(location)
    # Helper function to get next frame in sequence of iamges or video
    def get_next_frame(self):
        try:
            if self.type == 'image':
                imfile = self.images[FRAME_NUMBER]
                image = np.array(Image.open(imfile)).astype(np.uint8)
                image = cv2.resize(image, (320,240), interpolation = cv2.INTER_AREA)
                return image

            if self.type == 'video':
                _, image = self.images.read()
                image = cv2.resize(image, (320,240), interpolation = cv2.INTER_AREA)
                return image
        # In case reach end of sequence
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
        self.last_seen = 0
        self.veh_id = ID
        ID = ID + 1

    def update_full(self, frame_number, detection, flow):
        x, y, self.w, self.h = detection
        self.last_seen = 0
        self.x = self.x * 0.2 + x * 0.8 
        self.y = self.y * 0.2 + y * 0.8 
        self.x_dot, self.y_dot = flow

    def update_partial(self, flow):
        self.last_seen += 1
        self.x_dot = self.x_dot *0.2 +flow[0] *0.8
        self.y_dot = self.y_dot *0.2 +flow[1] *0.8

    def predict(self):
        self.x = self.x + self.x_dot
        self.y = self.y + self.y_dot

    def bounds(self):
        if self.last_seen > 3:
            return False

        if self.x < IMG_X_MAX and self.x > 0  and self.y > 0 and self.y < IMG_Y_MAX:
            return True
        else:
            return False
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
        full = 0
        initialize = 0
        partial = 0

        for i in range(len(self.bounding_boxes)):
            if min(error[i]) < 7.0:
                full += 1
                idx = int(np.where(error[i] == error[i].min())[0])
                self.measurement[self.prior_vehicles[idx].veh_id] = ['full',self.bounding_boxes[i], self.average_flow(self.bounding_boxes[i]), self.prior_vehicles[idx]]
            else:
                initialize += 1
                vehicle = Vehicle(self.frame_number, self.bounding_boxes[i], self.average_flow(self.bounding_boxes[i]))
                self.measurement[vehicle.veh_id] = ['initialize', 0, 0, vehicle]

        for vehicle in self.prior_vehicles:
            if vehicle.veh_id not in self.measurement:
                    partial += 1
                    bbox = [vehicle.x, vehicle.y, vehicle.w, vehicle.h]
                    self.measurement[vehicle.veh_id] = ['partial', 0, self.average_flow(bbox), vehicle]
        
        #print('Full: ' + str(full) + ' Partial: ' + str(partial) + ' New: ' + str(initialize))

    def update(self):
        for item in self.measurement.values():
            if item[0] == 'full':
                item[3].update_full(self.frame_number, item[1], item[2])
            if item[0] == 'partial':
                item[3].update_partial(item[2])
            if item[3].bounds():
                self.update_veh.append(item[3])

    def predict(self):
        for vehicle in self.update_veh:
            vehicle.predict()
            if vehicle.bounds():
                self.predict_veh.append(vehicle)

    def average_flow(self, bbox):
        x, y, w, h = bbox
        direction_x = np.average(self.optical_flow[int(y) : int(y +h) , int(x) : int(x+w), 0])
        direction_y = np.average(self.optical_flow[int(y) : int(y +h) , int(x):int(x+w) , 1])
        return [direction_x, direction_y]

    def get_bbox(self):
        bbox = []
        for vehicle in self.update_veh:
            bbox.append([int(vehicle.x), int(vehicle.y), int(vehicle.x + vehicle.w), int(vehicle.y + vehicle.h)])
        return bbox

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
            # load faster r-cnn
            self.detector = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            # send model to gpu
            self.detector.to(DEVICE)
            # set model to inference mode
            self.detector.eval()
            # set transformation to prepare image for network input
            self.transform = transforms.Compose([transforms.ToTensor()])
        if type == 'sinet':
            print('SINet')

    def inference(self, image):
        if self.type == 'frcnn':
            # convert image to torch tensor
            input = self.transform(image)
            # send input data to GPU
            input = input.to(DEVICE)
            # process inference and get detections
            detections = self.detector([input])
            boxes = detections[0]['boxes']
            confidence = detections[0]['scores']
            class_id = detections[0]['labels']



        if self.type == 'yolo':
            # convert image to torch tensor
            input = self.transform(image)
            # send input data to GPU
            input = input.to(DEVICE)
            # process inference and get detections
            detections = self.detector([input])
            boxes = detections[0]['boxes']
            confidence = detections[0]['scores']
            class_id = detections[0]['labels']

        if self.type == 'sinet':
            # convert image to torch tensor
            input = self.transform(image)
            # send input data to GPU
            input = input.to(DEVICE)
            # process inference and get detections
            detections = self.detector([input])
            boxes = detections[0]['boxes']
            confidence = detections[0]['scores']
            class_id = detections[0]['labels']


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
                if class_id[i] in [2,3,4,6, 7, 8]:
                    filter.append([int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])])
        return filter

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
class OpticalFlow:
    def __init__(self, type):
        self.type = type
        if type == 'farneback':

            print('Farneback')

        if type == 'raft':
            parser = argparse.ArgumentParser()
            parser.add_argument('--model', nargs='?', const='raft-models/raft-things.pth', type=str, help="restore checkpoint")
            parser.add_argument('--path', nargs='?', const='frames', type=int, help="dataset for evaluation")
            parser.add_argument('--small', action='store_true', help='use small model')
            parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
            parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
            args = parser.parse_args()
            args.model = 'raft-models/raft-things.pth'
            self.flow_model = torch.nn.DataParallel(RAFT(args))
            self.flow_model.load_state_dict(torch.load(args.model))
            self.flow_model = self.flow_model.module
            self.flow_model.to(DEVICE)
            self.flow_model.eval()

        if type == 'flownet':

            print('Flownet')


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
            self.result = flow_up[0].permute(1,2,0).detach().cpu().numpy()

        if self.type == 'flownet':
            self.flow = True

    def toimage(self):
        image = flow_viz.flow_to_image(self.result)
        return image
#---------------------------------------------------------------
# Helper functions
#---------------------------------------------------------------

# draw vehicle bounding box on input image
def draw_bbox(image, bboxes):
    copy = np.copy(image)
    for bbox in bboxes:
        cv2.rectangle(copy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), np.random.uniform(0, 255) , 2)
    return copy

# create optical flow + detection mask
def flow_mask(flow, bboxes):
    image = flow.toimage()
    mask = np.full(image.shape[:2], 0, dtype=np.uint8)
    for bbox in bboxes:
        cv2.rectangle(mask, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), -1)
    image = cv2.bitwise_or(np.array(image).astype(np.uint8), np.array(image).astype(np.uint8), mask=mask)
    return image

# create side by side images
def visuzalization(input, bbox, flow):
    bbox_img = draw_bbox(input, bbox)
    flow_visualization = flow.toimage()
    mask = flow_mask(flow, bbox)

    img1 = np.concatenate((input,bbox_img), axis = 1)
    img2 = np.concatenate((flow_visualization, mask), axis = 1)
    merge_img = np.concatenate((img1,img2), axis = 1)
    return merge_img

# update output image window
def display_result(image):
    cv2.imshow('image',image)
    cv2.waitKey(1)

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def evaluate(gt , result):
    iou_result = np.zeros((len(gt),len(result)), dtype=float)
    for i in range(len(gt)):
        for j in range(len(result)):
            box1 = [gt[i][2], gt[i][3], gt[i][4] , gt[i][5] ]
            box2 = [result[j][1], result[j][2], result[j][3] , result[j][4]]
            iou_result[i,j] = iou(box1, box2)
    tp = 0
    fn = 0
    fp = 0

    for i in range(len(gt)):
        if max(iou_result[i,:]) >= 0.7:
            tp += 1
        else:
            fn += 1
    
    for j in range(len(result)):
        if max(iou_result[:,j]) < 0.7:
            fp += 1
    
    return [tp , fn ,fp]


#-------------------------------------------------------------------------------
# INITIALIZE INPUT DATA --------------------------------------------------------
#-------------------------------------------------------------------------------
input = InputData(media_type, location)
current_frame = input.get_next_frame()
# INITIALIZE DETECTOR
detector = Detector(detector_type)
detector.inference(current_frame)
inital_veh = []
for detection in detector.result:
    vehicle = Vehicle(FRAME_NUMBER, detection, [0 ,0])
    inital_veh.append(vehicle)
# INITIALIZE Optical Flow
flow = OpticalFlow(flow_type)
#-------------------------------------------------------------------------------
if video_out:
    h, w = current_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_.mp4", fourcc, 5.0, (4*w, h))
#-------------------------------------------------------------------------------
if results_out:
    result_text = open(location + 'output.txt','w')
    # open csv file
#-------------------------------------------------------------------------------
if metrics_out:
    gt_text = open(location + 'gt.txt','r')
    gt_text = gt_text.readlines()
    gt = []
    for line in gt_text:
        data = line.split(',')
        gt.append([int(data[0]), int(data[1]), float(data[2]), float(data[3]), float(data[4])+ float(data[2]), float(data[5]) + float(data[3])])

    # open csv file
#-------------------------------------------------------------------------------
if time_out:
    times = []
#-------------------------------------------------------------------------------
# MAIN LOOP
#-------------------------------------------------------------------------------
while(current_frame is not False):
    #print('Frame No: ' + str(FRAME_NUMBER) + '  Veh. No.: ' + str(ID))
    # read Image
    if time_out:
        start = time.time()
    # get image pair
    previous_frame = current_frame
    current_frame  = input.get_next_frame()
    # check if reached end frame
    if current_frame is False:
        break
    # run detection
    detector.inference(current_frame)
    # run flow
    flow.inference(previous_frame, current_frame)
    # create frame
    if FRAME_NUMBER == 2: # first pair
        frame = Frame(detector.result, flow.result, inital_veh)
    else:
        frame = Frame(detector.result, flow.result, frame.predict_veh)
    # match
    frame.match()
    # update
    frame.update()
    # predict
    frame.predict()

    #############################################################################################################################
    # LOGGING RESULTS 
    #############################################################################################################################

    if time_out:
        processing_time = time.time()-start
        times.append(processing_time)
        #print('elapsed time: {}'.format(processing_time))

    if results_out:
        for vehicle in frame.predict_veh:
            line = str(frame.frame_number) + ',' + str(vehicle.veh_id) + ',' + str(int(vehicle.x)) +  ',' + str(int(vehicle.y)) + ',' + str(int(vehicle.w)) + ',' + str(int(vehicle.h)) + '\n'
            result_text.writelines((line))

    if metrics_out:
        expected = [item for item in gt if item[0] == FRAME_NUMBER]
        result = []
        for vehicle in frame.predict_veh:
            result.append([vehicle.veh_id , vehicle.x , vehicle.y, vehicle.w +vehicle.x , vehicle.h+vehicle.y])
        tp, tn , fp = evaluate(expected , result)
        performance[0] += tp
        performance[1] += tn
        performance[2] += fp
        #print(str(FRAME_NUMBER) + ' | TP : ' + str(tp) + ' | TN : ' + str(tn) + ' | FP : ' + str(fp))

    if video_out:
        image = visuzalization(current_frame, frame.get_bbox(), flow)
        out.write(image)

    if preview_result:
        image = visuzalization(current_frame, frame.get_bbox(), flow)
        display_result(image)

###################################################################################################################################
# HANDLING CLOSURES
###################################################################################################################################

if media_type == 'video':
    input.images.release()

if preview_result:
    cv2.destroyAllWindows()

if time_out:
    average = 1/np.average(times)
    print ('Average Frames Per Second: ' + str(average))

if metrics_out:
    precision =  performance[0] / (performance[0] + performance[2])
    recall = performance[0] / (performance[0] + performance[1])
    print ('Precision: ' + str(precision))
    print ('Recall: ' + str(recall))

if results_out:
    result_text.close()
