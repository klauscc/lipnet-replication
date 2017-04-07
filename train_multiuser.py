import numpy as np
import os
import sys
from gridDataset import GRIDDatasetGenerator 
from model.lipnet import * 
from keras.callbacks import ModelCheckpoint,CSVLogger


batch_size = 50
nb_epoch = 500
weight_savepath = './data/checkpoints/lipnet_weights_multiuser.hdf5'
log_savepath='./data/logs/lipnet_loss_seen_multiuser.csv'
log_savepath_unseen = './data/logs/lipnet_loss_unseen_multiuser.csv'

grid = GRIDDatasetGenerator()
net,test_func = lipnet(input_dim=grid.input_dim, output_dim=grid.output_dim, weights = weight_savepath)
#callbacks
checkpointer = ModelCheckpoint(filepath=weight_savepath,save_best_only=True,save_weights_only=True)
csv = CSVLogger(log_savepath)

nb_train_samples = 23000
#generators
train_gen = grid.next_train_batch(batch_size)
val_gen_seen = grid.next_val_batch(batch_size)
val_gen_unseen = grid.next_val_batch(batch_size, test_seen=False)


statisticCallback = StatisticCallback(test_func, log_savepath, val_gen_seen, 2850, weight_savepath)
statisticCallback_unseen = StatisticCallback(test_func, log_savepath_unseen, val_gen_unseen, 4000, None)
net.fit_generator(generator=train_gen, samples_per_epoch=nb_train_samples,
                    nb_epoch=nb_epoch,initial_epoch=0,
                    #validation_data=val_gen, nb_val_samples=nb_val_samples,
                    #callbacks=[checkpointer, csv, statisticCallback]
                    callbacks=[statisticCallback, statisticCallback_unseen]
                    )
