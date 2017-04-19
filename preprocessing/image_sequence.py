from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import os
import numpy as np
from tqdm import tqdm

class ImageSequenceGenerator(object):
    """generate image sequence"""
    def __init__(self):
        #hwc
        self.std = [0,0,0]
        self.mean = [0,0,0]

    def fit_generator(self, batch_generator, steps, save_file='./data/grid_channel_mean_std.npy'):
        if os.path.isfile(save_file):
            mean, std = np.load(save_file)
            self.mean = mean
            self.std = std 
            return
        x_mean = 0.
        xsquare_mean = 0.
        for i in tqdm(range(steps)):
            #nshwc x = next(batch_generator)[0].get('inputs')
            x_mean += 1. / (i+1) * (np.mean(x, axis=(0,1,2,3))-x_mean)
            xsquare_mean += 1. / (i+1) * (np.mean(x**2, axis=(0,1,2,3)) - xsquare_mean)
        self.mean = x_mean
        #var[x] = E[x^2] - (E[x])^2
        self.std = np.sqrt(xsquare_mean - x_mean**2)
        np.save(save_file, (self.mean, self.std))

    def random_deletion_duplication_frame(self, sequence, p=0.05):
        s,h,w,c = sequence.shape
        sequence_data = np.zeros([s,h,w,c])
        s_i=0
        for j in range(s):
            if np.random.rand() < p: 
                if np.random.rand() < 0.5: #perform duplication
                    sequence_data[s_i:s_i+2] = sequence[j,...]
                    s_i += 2
            else: #do not delete or duplication
                sequence_data[s_i] = sequence[j,...]
                s_i += 1
            if s_i >= s:
                break
        return sequence_data

    def standarize(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x
    """x must be a 4-rank array: (s,h,w,c)
    """
    def random_horizonal_flip(self, x, p=0.5):
        if np.random.rand() < p:
            return x[:,:,::-1,:]

    def flow(self, generator,x_key='inputs'):
        inputs,outputs = next(generator)
        #nshwc
        data_x = inputs[x_key]
        n,s,h,w,c = data_x.shape
        # for i in range(n):
            # for j in range(s):
                # data_x[i,j,...] = self.standarize(data_x[i,j,...])
            #perframe deletion or duplication by rate 0.05
            #data_x[i,...] = self.random_deletion_duplication_frame(data_x[i,...])
            #randomly horizonal flip
            # data_x[i,...] = self.random_horizonal_flip(data_x[i,...])
        return (inputs, outputs)
