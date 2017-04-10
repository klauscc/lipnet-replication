import os
import numpy as np
import h5py
import keras.backend as K
import glob
from PIL import Image as pil_image
from gridBaseDataset import GRIDBaseDataset

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
        np.random.seed(10^4)
        np.random.shuffle(train_lip_paths)
        train_n = len(train_lip_paths)
        split = 0.9
        train_num = int(train_n *split)
        test_unseen_paths = self.getLipPaths(self.test_people)

        self.train_paths=train_lip_paths[0:train_num]
        self.test_seen_paths=train_lip_paths[train_num:]
        self.test_unseen_paths=test_unseen_paths

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
