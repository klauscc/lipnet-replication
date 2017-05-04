import os
import numpy as np
import glob
import re
from preprocessing.MyImageDataGenerator import MyImageDataGenerator
from gridBaseDataset import GRIDBaseDataset

class GRIDPersonAuthentication(GRIDBaseDataset):
    def __init__(self, num_classes = 34, *args, **kwargs):
        GRIDBaseDataset.__init__(self, *args, **kwargs)

        self.timespecs = 1
        self.num_classes = num_classes
        people = np.arange(1,35)
        self.sample_paths = self.get_sample_paths(people, self.timespecs)
        #fix seed to keep the same splition of train dataset
        np.random.seed(10000)
        np.random.shuffle(self.sample_paths)
        np.random.seed()
        train_n = len(self.sample_paths)
        split_1 = 0.8
        split_2 = 0.9
        train_upper = int(train_n *split_1)
        test_upper = int(train_n * split_2)

        self.train_paths = self.sample_paths[0:train_upper]
        self.train_num = train_upper
        self.test_paths = self.sample_paths[train_upper:test_upper]
        self.test_num = test_upper - train_upper
        self.val_paths = self.sample_paths[test_upper:]
        self.val_num = train_n - test_upper

        self.standarized = False
        self.imageDataGenerator = MyImageDataGenerator(
            featurewise_center=True ,
            samplewise_center=False,
            featurewise_std_normalization=True,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=0.1,
            channel_shift_range=0.,
            fill_mode='nearest',
            horizontal_flip=True,
                )
        self.imageDataGenerator.fit_generator(self.next_batch(batch_size=32, phase="train", shuffle=True), self.train_num // 32)
        self.standarized = True

    def get_sample_paths(self, people, timespec=1, stride=None):
        paths = self.getLipPaths(people)
        sample_paths = []
        if not stride:
            stride = timespec
        per_sample = []
        for path in paths:
            for i in range(0, timespec, stride):
                if timespec == 1:
                    img = '{}/{}.jpg'.format(path, i)
                    if os.path.isfile(img):
                        sample_paths.append(img)
                else:
                    for j in range(i, i+timespec):
                        img = '{}/{}.jpg'.format(path, j)
                        if os.path.isfile(img):
                            per_sample.append(img)
                        else:
                            per_sample = []
                            break
                    if per_sample != []:
                        sample_paths.append(per_sample)
                        per_sample = []
        if self.shuffle:
            np.random.shuffle(sample_paths)
        return sample_paths

    def read_data(self,per_sample_path):
        def get_y_indice(one_path):
            match = re.match(r'.*\/s(\d+)\/.*',one_path)
            return int(match.group(1)) - 1

        y = np.zeros(self.num_classes)
        if type(per_sample_path) is list or type(per_sample_path) is tuple:
            x = np.zeros(len(per_sample_path), self.target_size[0], self.target_size[1], 3)
            y[get_y_indice(per_sample_path[0])] = 1
            for i, path in enumerate(per_sample_path):
                x[i,...] = self.load_image(path, target_size=self.target_size)
        else:
            x = self.load_image(per_sample_path, target_size=self.target_size)
            y[get_y_indice(per_sample_path)] = 1
        return x,y

    def gen_batch(self, begin, batch_size, paths):
        data = np.zeros([batch_size, self.target_size[0], self.target_size[1],3])
        label = np.zeros([batch_size, self.num_classes])
        for i in range(begin, begin+batch_size):
            x,y = self.read_data(paths[i])
            data[i-begin,...] = x
            label[i-begin,...] = y
        return (data, label)

    def next_batch(self, batch_size, phase, shuffle, random_transform=True):
        if phase == 'train':
            paths = self.train_paths
            num_sample = self.train_num
        elif phase == 'val':
            paths = self.val_paths
            num_sample = self.val_num
        else:
            paths = self.test_paths
            num_sample = self.test_num
        nb_steps = num_sample // batch_size
        while True:
            if shuffle:
                np.random.shuffle(paths)
            for itr in range(nb_steps):
                start_pos = itr*batch_size
                x,y = self.gen_batch(start_pos, batch_size, paths)
                x_transoformed = x.copy()
                if self.standarized == True and random_transform:
                    for i in range(x.shape[0]):
                        x_transoformed[i,...] = self.imageDataGenerator.random_transform(x[i,...])
                if self.debug:
                    print ("x shape: {}, y shape:{}".format(x_transoformed.shape, y))
                    print (x_transoformed[0], y[0])
                yield x_transoformed,y

if __name__ == "__main__":
    gridPersonDatabase = GRIDPersonAuthentication( debug = True)
    batch_size=100
    print ('gen a train batch.........')
    next(gridPersonDatabase.next_batch(batch_size, phase='train', shuffle = True))
    print ('gen a val batch.........')
    next(gridPersonDatabase.next_batch(batch_size, phase='val', shuffle = False))
