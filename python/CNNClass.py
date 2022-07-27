import numpy as np 
import pathlib

import seaborn as sns 
import matplotlib.pyplot as plt 

import tensorflow as tf  
from tensorflow.keras import layers
from tensorflow.keras import Model  
from tensorflow.keras.optimizers import RMSprop
from keras_preprocessing.image import ImageDataGenerator

from tensorflow import keras
from tensorflow.keras.models import Sequential


class CNN:
    #Image paramters
    batch_size = 10
    img_height = 200
    img_width = 250
    epochs = 10
    model = Sequential()
    class_names = ['hand', 'notHand']

    def saveModel(self, path):
        self.model.save(path)

    def createDataset(self,datasetPath):

        #image directories
        image_dir = data_dir = pathlib.Path(datasetPath)   #Setting image directory

        train_ds = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        validation_split=0.3,
        subset="training",
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size)

        self.class_names = train_ds.class_names

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds

    def defineCNN(self):

        resize_and_rescale = tf.keras.Sequential([
            layers.Resizing(self.img_width, self.img_height),
            layers.Rescaling(1./255)
        ])

        data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=(self.img_width,
                                        self.img_height,
                                        3)),
            layers.RandomRotation(0.5),
            layers.RandomZoom(0.8),
        ]
        )

        num_classes = len(self.class_names)

        self.model = Sequential([
            resize_and_rescale,
            data_augmentation,
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])


    def loadCNNModel(self,modelPath):
        self.model = tf.keras.models.load_model(modelPath)
        

    def trainCNN(self, train_set, test_set):
        
        self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        history = self.model.fit(train_set, validation_data=test_set, epochs=self.epochs)
        

    def predict(self, imagePath, scoreThreshold):
        img = tf.keras.utils.load_img(imagePath, target_size=(self.img_height, self.img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predictedClass = self.class_names[np.argmax(score)]
        return self.class_names[np.argmax(score)]
        

    def evaluateCNN(model, test_images, test_labels):
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        