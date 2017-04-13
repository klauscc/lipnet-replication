import os
import numpy as np
import h5py
import keras.backend as K
import glob
from PIL import Image as pil_image
from gridBaseDataset import GRIDBaseDataset
from preprocessing.image_sequence import *

class GRIDDatasetGenerator(GRIDBaseDataset):
    def __init__(self, test_people=(1,2,20,22), *args):
        GRIDBaseDataset.__init__(self, *args)
        self.test_people = test_people


        self.train_people = []
        for p in range(1,35):
            if p not in test_people:
                self.train_people.append(p)

        print ("train_people:{} test_people: {}".format(self.train_people, self.test_people))
        train_lip_paths = self.getLipPaths(self.train_people)

        #fix seed to keep the same splition of train dataset
        np.random.seed(10000)
        np.random.shuffle(train_lip_paths)
        np.random.seed()
        train_n = len(train_lip_paths)
        split = 0.9
        train_num = int(train_n *split)
        test_unseen_paths = self.getLipPaths(self.test_people)

        self.train_paths=train_lip_paths[0:train_num]
        self.test_seen_paths=train_lip_paths[train_num:]
        self.test_unseen_paths=test_unseen_paths

        self.train_num = train_num
        self.test_seen_num = len(self.test_seen_paths)
        self.test_unssen_num = len(self.test_unseen_paths)

        self.imageSequenceGenerator = ImageSequenceGenerator()
        self.imageSequenceGenerator.fit_generator(super(GRIDDatasetGenerator,self).next_train_batch(50), train_num//50)

    def next_train_batch(self, batch_size):
        while 1:
            yield self.imageSequenceGenerator.flow(super(GRIDDatasetGenerator, self).next_train_batch(batch_size))

    def next_val_batch(self, batch_size, test_seen=True):
        while 1:
            yield self.imageSequenceGenerator.flow(super(GRIDDatasetGenerator, self).next_val_batch(batch_size,test_seen))

class GRIDSingleUserDatasetGenerator(GRIDBaseDataset):
    def __init__(self, finetune_person=1, *args):
        GRIDBaseDataset.__init__(self, *args)

        self.train_people = []
        self.train_people.append(finetune_person)
        print ("fine tune person is {}".format(self.train_people))
        train_lip_paths = self.getLipPaths(self.train_people)

        train_n = len(train_lip_paths)
        split = 0.9
        train_num = int(train_n *split)
        self.train_paths=train_lip_paths[0:train_num]
        self.test_seen_paths=train_lip_paths[train_num:]

        default_unseen_people = (1,2,20,22)
        self.test_people = []
        for i in default_unseen_people:
            if i != finetune_person:
                self.test_people.append(i)
        self.test_unseen_paths = self.getLipPaths(self.test_people)
