import csv

from cv2 import IMREAD_ANYCOLOR
from CNNClass import *
from YOLO import *
import cv2


#ActiveDir used as connection point with the C++ modulebeh 
activeDir = "../activeDir/"

# Initialize and load YOLO model
#The directory is set w.r.t. the CMAKE file starting point 
#if the script is launched throuhg the c++ module

yolo = YOLO('../models/yolo/yoloTrainedEgo.cfg',
            '../models/yolo/yoloTrainedEgo.weights', ["hand"])


def loadHaar(path):
    detector = cv2.CascadeClassifier(path)
    return detector

#Load 
def loadMainModel(mainModelPath, cnn):
    cnn.loadCNNModel(mainModelPath)


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

#Calls the inference YOLO method to perfom the detection
def detectROI(imge):
    results = yolo.inference(imge)
    return results

#Writes the coordinates into the CSV file 
def writeCsv(savePath, rows):
    # writing to csv file
    with open(savePath, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_NONE)
        # writing the data rows
        csvwriter.writerows(rows)

#Perfomrs the detection and writes the results into the ActiveDir defined above
def detect(srcPath):
    src = cv2.imread(srcPath)
    # detect ROI for hands for boundingBoxes
    handRoi = detectROI(src)
    rows = []
    for detection in handRoi:
        x, y, w, h = detection
        handFrame = src[x:x+w, y:y+h]
        savepath = activeDir + "test.jpg"
        #predicetedSrc = cnn.predict(savepath)
        # print(predicetedSrc)
        rows.append([x, y, w, h])
    writeCsv(activeDir + "test.csv", rows)

#Python Main entry point
if __name__ == "__main__":
    srcPath = activeDir + "src.jpg"
    detect(srcPath)
    print("done")
    quit()
