from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from tqdm import tqdm

class MyImageDataGenerator(ImageDataGenerator):
    """docstring for MyImageDataGenerator"""
    def __init__(self, *args, **kwargs):
        super(MyImageDataGenerator, self).__init__(*args, **kwargs)
        
    def fit_generator(self, generator, steps, save_file='./data/image_grid_channel_mean_std.npy'):

        if os.path.isfile(save_file):
            mean, std = np.load(save_file)
            self.mean = mean
            self.std = std 
            return
        x_mean = [0.,0., 0.]
        xsquare_mean = [0.,0., 0.]
        for i in tqdm(range(steps)):
            #nhwc 
            x = next(generator)[0]
            x_mean += 1. / (i+1) * (np.mean(x, axis=(0, self.row_axis, self.col_axis))-x_mean)
            xsquare_mean += 1. / (i+1) * (np.mean(x**2, axis=(0,self.row_axis, self.col_axis)) - xsquare_mean)
        self.mean = x_mean
        #var[x] = E[x^2] - (E[x])^2
        self.std = np.sqrt(xsquare_mean - x_mean**2)
        print ("data mean:{}, std:{}".format(self.mean, self.std))
        np.save(save_file, (self.mean, self.std))

    def preprocess(self,x, do_transform=False):
        if do_transform:
            y = np.empty_like(x)
            y = self.random_transform(x)
            return self.standarized(y)
        else:
            return self.standarized(x)

    def preprocess_batch(self,x,do_transform=False):
        n = x.shape[0]
        x_preprocessed = np.copy(x)
        # for i in range(n):
            # if do_transform: 
                # x_preprocessed[i,...] = self.random_transform(x[i])
            # x_preprocessed[i,...] = self.standardize(x_preprocessed[i,...])
        return x_preprocessed
