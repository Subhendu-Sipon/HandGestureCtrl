import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier

img = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

padding = 25
imgSize = 300

labels = ["Stop", "Start"]


while True:
    ret, frame = img.read()
    hands , frame = detector.findHands(frame)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize,3) , np.uint8) * 255

        imgCrop = frame[y-padding:y+h+padding ,x-padding:x+w+padding]
        
        imgCropShape = imgCrop.shape

        
        aspectRatio = h/w

        if aspectRatio >1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
        
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)


        print(prediction, index)
        cv.imshow('Cropped Hand', imgCrop)
        cv.imshow('White Image', imgWhite)


    cv.imshow('Webcam Opened', frame)
    cv.waitKey(1)
    