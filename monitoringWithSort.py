from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
import time
from sort import Sort

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

#tracking
tracker = Sort(max_age = 20, min_hits=3, iou_threshold=0.3)

limitsUp = [210, 450, 600, 450]
limitsDown = [650, 450, 1000, 450]

totalCountUp = []
totalCountDown = []

clsCounterUp = {'car' : 0, 'truck' : 0, 'motorbike': 0}  
clsCounterDown = {'car' : 0, 'truck' : 0, 'motorbike': 0}

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    imgRegion = cv2.bitwise_and(img, mask)
    img = cvzone.overlayPNG(img, mainCounter, (300, 0))
    img = cvzone.overlayPNG(img, outCounter, (0, 0))
    img = cvzone.overlayPNG(img, inCounter, (880, 0))
    results = model(imgRegion, stream = True)
    detections = np.empty((0,6))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # BBOX
            print(box)
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
            if currentClass == 'car':
                # cvzone.putTextRect(img, text = f"{classNames[cls]} {conf}", pos=(max(0, x1), max(35, y1)),
                #                 scale = 0.6,
                #                 thickness=1, 
                #                 offset=3)
                #cvzone.cornerRect(img, bbox = bbox, l= 15 , rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf, cls])
                detections = np.vstack((detections, currentArray))

            if currentClass == "truck":
                currentArray = np.array([x1, y1, x2, y2, conf, cls])
                detections = np.vstack((detections, currentArray))

            if currentClass == "motorbike":
                currentArray = np.array([x1, y1, x2, y2, conf, cls])
                detections = np.vstack((detections, currentArray))
            

    # print("printing directions")
    # print(detections)
    classes_array = detections[:,-1:]
    # print(classes_array)
    resultsTracker = tracker.update(detections)
    # print("printing result tracker")
    # print(resultsTracker)
    # print(resultsTracker.shape, classes_array.shape)
    try:
        resultsTracker = np.hstack((resultsTracker, classes_array))
    except ValueError:
        classes_array = classes_array[:resultsTracker.shape[0], :]
        resultsTracker = np.hstack((resultsTracker, classes_array))
    # print(resultsTracker)
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), thickness=5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), thickness=5)

    for result in resultsTracker:
        x1, y1, x2, y2, id, cls = result
        x1, y1, x2, y2, id, cls = int(x1), int(y1), int(x2), int(y2), int(id), int(cls)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt = 2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, text = f"{model.names[cls]} {id}", pos=(max(0, x1), max(35, y1)),
                scale = 2,
                thickness=3, 
                offset=10)
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), radius = 5, color= (255, 0, 255), thickness= cv2.FILLED)
        currentClass = model.names[cls]
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                clsCounterUp[currentClass] += 1
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255 , 0), thickness=3)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                clsCounterDown[currentClass] += 1
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255 , 0), thickness=3)

    cv2.putText(img, str(len(totalCountUp)), (565, 112), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(len(totalCountDown)), (750, 112), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(clsCounterUp["car"]), (95, 92), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(clsCounterUp["truck"]), (95, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(clsCounterUp["motorbike"]), (95, 146), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(clsCounterDown["car"]), (1150, 92), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(clsCounterDown["truck"]), (1150, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(clsCounterDown["motorbike"]), (1150, 146), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    # cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
