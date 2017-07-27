import numpy as np
import os
import sys
from gridDatasetGenerator import GRIDDatasetGenerator 
from model.lipnet import * 
from model.lipnet_res3d import lipnet_res3d
from keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from time import gmtime, strftime
from configurations import init

init()

batch_size = 32
nb_epoch = 500
name =  'multiuser_with_auth'
pre_weights = ''
initial_epoch = 0
timestamp=strftime("%Y_%m_%d__%H_%M_%S",gmtime())
weight_savepath = './data/checkpoints/lipnet_res3d_weights_{}_{}.hdf5'.format(name, timestamp)
log_savepath='./data/logs/lipnet_res3d_loss_seen_{}_{}.csv'.format(name, timestamp)
log_savepath_unseen = './data/logs/lipnet_res3d_loss_unseen_{}_{}.csv'.format(name, timestamp)

grid = GRIDDatasetGenerator()
net,test_func = lipnet_res3d(grid.input_dim, grid.output_dim, weights=None) 
net.summary()
if pre_weights:
    net.load_weights(pre_weights) 
#callbacks
checkpointer = ModelCheckpoint(filepath=weight_savepath,save_best_only=True,save_weights_only=True)
csv = CSVLogger(log_savepath)
nb_train_samples = grid.train_num
#generators
train_gen = grid.next_train_batch(batch_size, gen_words=False)
val_gen_seen = grid.next_val_batch(batch_size, gen_words=False)
val_gen_unseen = grid.next_val_batch(batch_size, test_seen=False, gen_words=False)

statisticCallback = StatisticCallback(test_func, log_savepath, val_gen_seen, grid.test_seen_num, weight_savepath)
statisticCallback_unseen = StatisticCallback(test_func, log_savepath_unseen, val_gen_unseen, grid.test_unssen_num, None)
net.fit_generator(generator=train_gen, 
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=nb_epoch,initial_epoch=initial_epoch,
                    validation_data=val_gen_seen, validation_steps=grid.test_seen_num // batch_size,
                    callbacks=[statisticCallback, statisticCallback_unseen]
                    )
