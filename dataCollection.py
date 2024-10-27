import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

img = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)

padding = 25
imgSize = 300

folder = 'Data/Stop'
counter = 0

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
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize



        cv.imshow('Cropped Hand', imgCrop)
        cv.imshow('White Image', imgWhite)


    cv.imshow('Webcam Opened', frame)
    key = cv.waitKey(1)
    if key == ord('s'):
        counter+=1
        cv.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)