import zipfile
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Rescaling


import config

class data:
    
    def __init__(self):
        self.classes=[]
        self.train_data=None
        self.test_data=None
        
    def get_data(self):
        if not os.path.exists(config.DATA_DIR) or len(os.listdir(config.DATA_DIR))==0:
            os.makedirs(config.DATA_DIR,exist_ok=True)
            print(f'There no zip file with path {config.ZIPFILE_PATH} ')
            print(f'Download the dataset zip file format from : "https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product" \n and place this zip file in path : "{config.DATA_DIR}/" , if the zipfile name is other than "archive (6).zip" then rename it  ')
            return
        with zipfile.ZipFile(config.ZIPFILE_PATH,'r') as file:
             file.extractall(config.DATA_DIR)
             
        #load data
        self.train_data=tf.keras.utils.image_dataset_from_directory(
            config.TRAINING_DATA,
            labels='inferred',
            image_size=(128,128),
            shuffle=True,
            batch_size=32,
            seed=32
        )
        self.test_data=tf.keras.utils.image_dataset_from_directory(
            config.TESTING_DATA,
            labels='inferred',
            image_size=(128,128),
            shuffle=True,
            batch_size=32,
            seed=32
        )
        self.classes=self.train_data.class_names
        
        #scaling
        def scale_images(image,label):
            rescaler=Rescaling(1./255)
            image=rescaler(image)
            return image,label
        
        self.scaled_train=self.train_data.map(scale_images)
        self.scaled_test=self.test_data.map(scale_images)
        
        data_gen=ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,   
        )
    
    def plot_sample(self):
        """
        plot sample images from train data
        """
        if self.train_data is None:
            raise ValueError("Train data is not loaded. Call `load_data()` first.")
        os.makedirs(config.PLOTS_PATH,exist_ok=True)
        plt.figure(figsize=(9,9))
        for image,label in self.train_data.take(1):
            for i in range(9):
                plt.subplot(3,3,i+1)
                plt.imshow(image[i].numpy().astype('uint8'))
                plt.title(self.classes[label[i]])
                plt.axis('off')
        plt.savefig(config.PLOTS_PATH+'/sample_plot.png')
        plt.show()
            
        
if __name__=='__main__':
    dt=data()
    dt.get_data()
    dt.plot_sample()