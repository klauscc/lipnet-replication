import os
import numpy as np
import h5py
import keras.backend as K
import glob
from PIL import Image as pil_image

class GRIDBaseDataset(object):
    def __init__(self, target_size=[50,100], shuffle=True, re_generate=False ,data_dir='./data/GRID', dst_path='./data/grid_hkl/GRID.h5'):
        self.data_root = data_dir
        self.lip_dir = data_dir+'/lip'
        self.label_dir = data_dir+'/alignments'
        self.dataset_path = dst_path
        self.re_generate = re_generate
        self.timespecs = 75
        self.ctc_blank = 27
        self.target_size = target_size
        self.max_label_length = 50
        self.shuffle=shuffle

        self.input_dim = (self.timespecs, target_size[0], target_size[1], 3)
        self.output_dim = (self.max_label_length)

        self.train_paths = None
        self.test_seen_paths = None
        self.test_unseen_paths = None

    """
    generate next training batch
    """
    def next_train_batch(self, batch_size):
        nb_iterate = len(self.train_paths) // batch_size
        while True:
            if self.shuffle:
                np.random.shuffle(self.train_paths)
            for itr in range(nb_iterate):
                start_pos = itr*batch_size
                yield self.gen_batch(start_pos, batch_size, self.train_paths)

    """
    generate next validation batch
    """
    def next_val_batch(self, batch_size, test_seen=True):
        if test_seen:
            paths = self.test_seen_paths
        else:
            paths = self.test_unseen_paths

        nb_iterate = len(paths) // batch_size
        while True:
            for itr in range(nb_iterate):
                start_pos = itr*batch_size
                yield self.gen_batch(start_pos, batch_size, paths)

    def gen_batch(self, begin, batch_size, paths):
        data = np.zeros([batch_size, self.timespecs, self.target_size[0], self.target_size[1], 3])
        label = np.zeros([batch_size, self.max_label_length])
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        source_strs = []

        for i in range(batch_size):
            pos = begin+i
            lip_d, lip_l, lip_label_len, source_str = self.readLipSequences(paths[pos])
            data[i] = lip_d
            label[i] = lip_l
            input_length[i] = self.timespecs - 2
            label_length[i] = lip_label_len
            source_strs.append(source_str)

        inputs = {'inputs': data,
                'labels': label,
                'input_length': input_length,
                'label_length': label_length,
                'source_str': source_strs
                }
        outputs = {'ctc':np.zeros([batch_size])}
        return (inputs, outputs)

    """
    convert a word comprised of characters to an tuple of integer
    a -> 0
    b -> 1
    z -> 25

    example:
    one -> (14, 13, 4)
    """
    def convertWordToLabel(self, string, padding_blank=False):
        if padding_blank:
            label = [self.ctc_blank]
        else:
            label = []
        for char in string:
            label.append(ord(char) - ord('a'))
        if padding_blank:
            label.append(self.ctc_blank)
        return label

    """
    convert align file to label:

    an align file looks like:

    ```
    0 23750 sil
    23750 29500 bin
    29500 34000 blue
    34000 35500 at
    35500 41000 f
    41000 47250 two
    47250 53000 now
    53000 74500 sil
    ```
    so the word list is (bin, blue, at, f, two, now) then convert it to interger tuple.

    """
    def convertAlignToLabels(self, align_file):
        with open(align_file,'r') as f:
            lines = f.readlines()
            words = []
            frames = []
            sentence_label = []
            source_str = ''
            for i,line in enumerate(lines):

                #remove first and last SIL word
                if i ==0:
                    continue
                if i == len(lines)-1:
                    continue

                striped_line = line.rstrip()
                begin,end,word = striped_line.split(' ')
                source_str += word
                begin_frame = int(begin) // 1000 - 1
                end_frame = int(end) // 1000 - 1
                words.append(self.convertWordToLabel(word, padding_blank=True))
                frames.append([begin_frame, end_frame])
                sentence_label.extend(self.convertWordToLabel(word))
                if i!=len(lines)-1:
                    sentence_label.append(26)
                    source_str += ' '
            label = np.zeros(self.max_label_length)
            label -= 1
            label[0:len(sentence_label)] = sentence_label
            label_len = len(sentence_label)
        return (label, label_len, words, frames, source_str)

    """
    load an image and convert each pixel value to range of (-1,1)
    """
    def load_image(self, path, grayscale=False, target_size=None):
        img = pil_image.open(path)
        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        if target_size:
            img = img.resize((target_size[1],target_size[0]))
        img = np.asarray(img, dtype=float)
        return self.preprocess_input(img)

    def preprocess_input(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    """
    read a lip sequence to numpy array
    """
    def readLipSequences(self, lipsequence_dir):
        total = len(sum([i[2] for i in os.walk(lipsequence_dir)],[]))
        sequence_name = lipsequence_dir.split('/')[-1]
        sequence_owner = lipsequence_dir.split('/')[-2]
        lip_sequence = np.zeros([self.timespecs, self.target_size[0], self.target_size[1], 3])
        for i in range(total):
            img_name = '{}/{}.jpg'.format(lipsequence_dir, i)
            lip_sequence[i,...] = self.load_image(img_name, target_size=self.target_size)
        label_path = self.getAlignmentDirOfPerson(sequence_owner, sequence_name)
        sentence_label, label_length, words, frames, source_str = self.convertAlignToLabels(label_path)
        return (lip_sequence, sentence_label, label_length, source_str)

    def getLipDirOfPerson(self, i):
        return "{}/lip/s{}".format(self.data_root, i)

    def getAlignmentDirOfPerson(self, i, name):
        return "{}/alignments/{}/align/{}.align".format(self.data_root, i, name)

    def getLipPaths(self, people):
        paths = []
        for i in people:
            person_lip_dir = self.getLipDirOfPerson(i)
            lip_dirs = glob.glob(person_lip_dir+'/*')
            paths.extend(lip_dirs)
        return paths

    """
    write the the samples in list `lip_paths` to a hdf5 file
    """
    def writeToH5(self, f, dset_data_name, dset_label_name, lip_paths):
        dset_data = f.create_dataset(dset_data_name,(train_num, self.timespecs, self.target_size[0], self.target_size[1], 3))
        dset_label = f.create_dataset(dset_data_name,(train_num, self.max_label_length))
        for i, path in enumerate(lip_paths):
            print "generating '{}'...".format(path)
            data, label = self.readLipSequences(path)
            dset_data[i,...] = data
            dset_label[i,...] = label
    """
    save all the samples to a hdf5 file
    the disadvantage is samples cannot shuffle at each epoch
    """
    def gen_hdf5(self):
        f = h5py.File(self.dataset_path,'w')

        #train and test_unseen dataset
        train_lip_paths = self.getLipPaths(self.train_people)
        np.random.shuffle(train_lip_paths)
        train_n = len(train_lip_paths)
        split = 0.9
        train_num = int(train_n *split)
        self.writeToH5(f, 'train_data', 'train_label', train_lip_paths[0:train_num])
        self.writeToH5(f, 'test_seen_data', 'test_seen_lable', train_lip_paths[train_num:])
        #test unseen dataset
        test_unseen_paths = self.getLipPaths(self.test_people)
        np.random.shuffle(test_unseen_paths)
        self.writeToH5(f, 'test_unseen_data', 'test_unseen_label', test_unseen_paths)

        f.close()

