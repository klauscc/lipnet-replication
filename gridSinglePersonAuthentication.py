import os
import numpy as np
import glob
import re
from preprocessing.MyImageDataGenerator import MyImageDataGenerator
from gridBaseDataset import GRIDBaseDataset
from keras.preprocessing.image import * 
import logging

class GRIDSinglePersonAuthentication(GRIDBaseDataset):
    def __init__(self, auth_person=23, pos_n=25, *args, **kwargs):
        GRIDBaseDataset.__init__(self, *args, **kwargs)
        self.num_classes = 2
        self.auth_person = auth_person
        neg_people = np.arange(1,20)
        logging.info( 'getting positive and negative lip paths...') 
        self.neg_lip_paths = self.getLipPaths(neg_people) 
        self.pos_lip_paths = self.getLipPaths( (auth_person, ) ) 
        #fix seed to keep the same splition of train dataset
        np.random.seed(100)
        np.random.shuffle(self.pos_lip_paths) 
        np.random.shuffle(self.neg_lip_paths) 
        np.random.seed()

        self.pos_train_paths = self.pos_lip_paths[0:pos_n] 
        self.pos_val_paths = self.pos_lip_paths[pos_n:] 
        split = int(len(self.neg_lip_paths)*0.9)
        self.neg_train_paths = self.neg_lip_paths[0:split] 
        other_people = np.arange(21,34) 
        unseen_neg_people = [] 
        for person in other_people:
            if person != auth_person:
                unseen_neg_people.append(person) 
        # unseen_neg_people = (1,2,3,4) 
        self.neg_val_paths = self.getLipPaths(unseen_neg_people ) 
        np.random.shuffle(self.neg_val_paths) 

    def next_batch(self, batch_size, phase, gen_words=False, shuffle=False):
        if phase is 'train':
            pos_paths = self.pos_train_paths
            neg_paths = self.neg_train_paths
        elif phase is 'val':
            pos_paths = self.pos_val_paths
            neg_paths = self.neg_val_paths
        else:
            raise ValueError( 'phase must be one of {train, val}') 
        pos_n  = len(pos_paths) 
        nb_iterate = pos_n // batch_size + 1
        while True:
            if self.shuffle:
                np.random.shuffle(pos_paths)
            for itr in range(nb_iterate):
                start_pos = itr*batch_size
                current_batch_size = batch_size
                if pos_n < start_pos + batch_size//2:
                    current_batch_size = (pos_n - start_pos) * 2
                paths = []
                paths.extend(pos_paths[start_pos: start_pos+current_batch_size//2] )
                paths.extend(np.random.choice(neg_paths, size= current_batch_size // 2, replace=False))
                if self.debug:
                    logging.info("iteration: {}. paths:{}".format(itr, paths) ) 
                yield self.gen_batch(0, current_batch_size, paths, gen_words=gen_words, auth_person=self.auth_person, scale=1/255.)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)  
    gridPersonDatabase = GRIDSinglePersonAuthentication( debug = True)
    batch_size=100
    print ('gen a train batch.........')
    next(gridPersonDatabase.next_batch(batch_size, phase='train', shuffle = True))
    print ('gen a val batch.........')
    next(gridPersonDatabase.next_batch(batch_size, phase='val', shuffle = False))
