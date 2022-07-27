from email.mime import image
from multiprocessing.connection import wait
import csv

from cv2 import imread, imshow, waitKey
from matplotlib.image import imsave
from CNNClass import *
from YOLO import *
import sys
import cv2
from imutils.video import VideoStream
import argparse
import imutils
import time
import numpy as np
import os

cnn = CNN()

activeDir = "../activeDir/"

yolo = YOLO('../python/models/yolo/yoloTrained.cfg',
            '../python/models/yolo/yoloTrained.weights', ["hand"])


def loadHaar(path):
    detector = cv2.CascadeClassifier(path)
    return detector


def loadMainModel(mainModelPath):
    cnn.loadCNNModel(mainModelPath)


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img



def detectROI(imge):
    results = yolo.inference(imge)
    return results

def writeCsv(savePath, rows):        
    # writing to csv file 
    with open(savePath, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_NONE)
        # writing the data rows 
        csvwriter.writerows(rows)


def detect(srcPath, scoreThreshold):
    src = imread(srcPath)
    #detect ROI for hands for boundingBoxes
    handRoi = detectROI(src)
    rows = []
    for detection in handRoi:
        x, y, w, h = detection
        handFrame = src[x-100:x+w+100, y-100:y+h+100]
        savepath = activeDir + "test.jpg"
        imsave(savepath, handFrame)
        predicetedSrc = cnn.predict(savepath, scoreThreshold)
        print(predicetedSrc)
        rows.append([x,y,w,h])
    writeCsv(activeDir + "test.csv", rows)



if __name__ == "__main__":
    cnn.defineCNN()
    loadMainModel("../python/models/CNN")
    scoreThreshold = 0.9
    srcPath = activeDir + "src.jpg"
    detect(srcPath, scoreThreshold)
    print("done")
    quit()
