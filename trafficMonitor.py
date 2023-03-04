from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import get_class_color, estimatedSpeed
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

#cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("./Videos/traffic1.mp4") # For video

model  = YOLO("./YoloWeights/yolov8l.pt") #large model works better with the GPU

mask = cv2.imread("static/mask.png")

mainCounter = cv2.imread("static/main_counter.png", cv2.IMREAD_UNCHANGED)
mainCounter = cv2.resize(mainCounter, (700, 250))
outCounter = cv2.imread("static/out.png", cv2.IMREAD_UNCHANGED)
inCounter = cv2.imread("static/in.png", cv2.IMREAD_UNCHANGED)

#aboutDeveloper = cv2.imread("static/about_developer.png", cv2.IMREAD_UNCHANGED)
#aboutDeveloper = cv2.resize(aboutDeveloper, (300, 90))
#tracking
tracker = DeepSort(
    max_iou_distance=0.7,
    max_age=2,
    n_init=3,
    nms_max_overlap=3.0,
    max_cosine_distance=0.2)

limitsUp = [210, 450, 600, 450]
limitsDown = [650, 450, 1000, 450]

totalCountUp = []
totalCountDown = []

coordinatesDict = dict()

clsCounterUp = {'car' : 0, 'truck' : 0, 'motorbike': 0}  
clsCounterDown = {'car' : 0, 'truck' : 0, 'motorbike': 0}

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    imgRegion = cv2.bitwise_and(img, mask)
    img = cvzone.overlayPNG(img, mainCounter, (300, 0))
    img = cvzone.overlayPNG(img, outCounter, (0, 0))
    img = cvzone.overlayPNG(img, inCounter, (880, 0))
    #img = cvzone.overlayPNG(img, aboutDeveloper, (980, 610))
    results = model(imgRegion, stream = True)
    detections = list()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            bbox = (x1, y1, w, h)
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            
            # Class Name
            cls = int(box.cls[0])

            currentClass = model.names[cls]
            if currentClass == 'car' and conf > 0.5:
                w, h = x2 - x1, y2 - y1 
                detections.append(([x1, y1, w, h], conf, cls))

            elif currentClass == "truck":
                w, h = x2 - x1, y2 - y1 
                detections.append(([x1, y1, w, h], conf, cls))

            elif currentClass == "motorbike":
                w, h = x2 - x1, y2 - y1 
                detections.append(([x1, y1, w, h], conf, cls))
            
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), thickness=5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), thickness=5)
    
    tracks = tracker.update_tracks(detections, frame=img)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id

        bbox = track.to_ltrb()
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        w, h = x2 - x1, y2 - y1 
        
        co_ord = [x1, y1]

        if track_id not in coordinatesDict:
            coordinatesDict[track_id] = co_ord
        else :
            if len(coordinatesDict[track_id]) > 2: 
                del coordinatesDict[track_id][-3:-1]
            coordinatesDict[track_id].append(co_ord[0])
            coordinatesDict[track_id].append(co_ord[1])
        estimatedSpeedValue = 0
        if len(coordinatesDict[track_id]) > 2:
            location1 = [coordinatesDict[track_id][0], coordinatesDict[track_id][2]]
            location2 = [coordinatesDict[track_id][1], coordinatesDict[track_id][3]]
            estimatedSpeedValue = estimatedSpeed(location1, location2)

        cls = track.get_det_class()
        currentClass = model.names[cls]
        clsColor = get_class_color(currentClass)

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt = 2, colorR=clsColor)

        cvzone.putTextRect(
                img, 
                text = f"{model.names[cls]} {estimatedSpeedValue} km/h",
                pos=(max(0, x1), max(35, y1)),
                colorR= clsColor,
                scale = 1,
                thickness=1, 
                offset=2)
        
        cx, cy = x1+w//2, y1+h//2

        cv2.circle(img, (cx, cy), radius = 5, color= clsColor, thickness= cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(track_id) == 0:
                totalCountUp.append(track_id)
                clsCounterUp[currentClass] += 1
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (255, 255 , 255), thickness=3)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(track_id) == 0:
                totalCountDown.append(track_id)
                clsCounterDown[currentClass] += 1
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (255, 255 , 255), thickness=3)

    cv2.putText(img, str(len(totalCountUp)), (565, 112), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(len(totalCountDown)), (750, 112), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(clsCounterUp["car"]), (95, 92), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(clsCounterUp["truck"]), (95, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(clsCounterUp["motorbike"]), (95, 146), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(clsCounterDown["car"]), (1150, 92), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(clsCounterDown["truck"]), (1150, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(clsCounterDown["motorbike"]), (1150, 146), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow('Traffic Monitoring', img)
    cv2.waitKey(1)
