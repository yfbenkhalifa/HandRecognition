from CNNClass import *

cnn = CNN()
train, val = cnn.createDataset("dataset/processed/")
cnn.defineCNN()
cnn.trainCNN(train,val)
cnn.saveModel("python/models/CNN")