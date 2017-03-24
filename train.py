import numpy as np
import os
import sys
from gridDataset import GRIDDatasetGenerator 
from model.lipnet import lipnet 
from keras.callbacks import ModelCheckpoint,CSVLogger


batch_size = 50
nb_epoch = 50
weight_savepath = './data/checkpoints/lipnet_weights.hdf5'
log_savepath='./data/logs/lipnet_loss.csv'

grid = GRIDDatasetGenerator()
net = lipnet(input_dim=grid.input_dim, output_dim=grid.output_dim, weights = weight_savepath)
#callbacks
checkpointer = ModelCheckpoint(filepath=weight_savepath,save_best_only=True,save_weights_only=True)
csv = CSVLogger(log_savepath)
#generators
train_gen = grid.next_train_batch(batch_size)
val_gen = grid.next_val_batch(batch_size)

net.fit_generator(generator=train_gen, samples_per_epoch=23000,
                    nb_epoch=nb_epoch, validation_data=val_gen, nb_val_samples=2850,
                    callbacks=[checkpointer, csv])
