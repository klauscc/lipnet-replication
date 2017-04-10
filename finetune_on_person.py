import numpy as np
import os
import sys
from gridDatasetGenerator import GRIDSingleUserDatasetGenerator 
from model.lipnet import * 
from keras.callbacks import ModelCheckpoint,CSVLogger


batch_size = 50
nb_epoch = 100
finetune_person = 1
weight_savepath = './data/checkpoints/lipnet_weights_user_{}.hdf5'.format(finetune_person)
log_savepath='./data/logs/lipnet_loss_seen_user_{}.csv'.format(finetune_person)
log_savepath_unseen = './data/logs/lipnet_loss_unseen_user_{}.csv'.format(finetune_person)

grid = GRIDSingleUserDatasetGenerator(finetune_person=finetune_person)
net,test_func = lipnet(input_dim=grid.input_dim, output_dim=grid.output_dim, weights = './data/checkpoints/lipnet_weights_multiuser_backend.hdf5')
#callbacks
checkpointer = ModelCheckpoint(filepath=weight_savepath,save_best_only=True,save_weights_only=True)

nb_train_samples = 400
#generators
train_gen = grid.next_train_batch(batch_size)
val_gen_seen = grid.next_val_batch(batch_size)
val_gen_unseen = grid.next_val_batch(batch_size, test_seen=False)


statisticCallback = StatisticCallback(test_func, log_savepath, val_gen_seen, 100, weight_savepath)
statisticCallback_unseen = StatisticCallback(test_func, log_savepath_unseen, val_gen_unseen, 3000, None)
net.fit_generator(generator=train_gen, steps_per_epoch=18,
                    nb_epoch=nb_epoch,initial_epoch=0,
                    #callbacks=[statisticCallback, statisticCallback_unseen]
                    callbacks=[statisticCallback]
                    )
