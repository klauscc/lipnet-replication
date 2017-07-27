import numpy as np
import os
import sys
from gridDatasetGenerator import GRIDDatasetGenerator 
from model.lipnet import * 
from keras.callbacks import ModelCheckpoint,CSVLogger
from time import gmtime, strftime
from configurations import init

init()

batch_size = 50
nb_epoch = 500
weight_savepath = ''
timestamp=strftime("%Y_%m_%d__%H_%M_%S",gmtime())
log_savepath='./data/logs/lipnet_auth_{}.csv'.format(timestamp)
log_savepath_unseen = './data/logs/lipnet_auth_unseen_{}.csv'.format(timestamp)

grid = GRIDDatasetGenerator()
net = authnet(input_dim=grid.input_dim, output_dim=grid.output_dim, weights = weight_savepath)
#callbacks
checkpointer = ModelCheckpoint(filepath='./data/checkpoints/lipnet_auth.hdf5',save_best_only=True,save_weights_only=True)
csv = CSVLogger(log_savepath)

nb_train_samples = grid.train_num
nb_val_samples = grid.test_seen_num
#generators
train_gen = grid.next_train_batch(batch_size, gen_words=False)
val_gen_seen = grid.next_val_batch(batch_size, gen_words=False)
val_gen_unseen = grid.next_val_batch(batch_size, test_seen=False, gen_words=False)

net.fit_generator(generator=train_gen, steps_per_epoch=nb_train_samples // batch_size,
                    epochs=nb_epoch,initial_epoch=0,
                    validation_data=val_gen_seen, nb_val_samples=nb_val_samples,
                    callbacks=[checkpointer, csv]
                    # callbacks=[statisticCallback, statisticCallback_unseen]
                    )
