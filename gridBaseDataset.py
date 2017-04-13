import os
import numpy as np
import h5py
import keras.backend as K
import glob
from PIL import Image as pil_image



class GRIDBaseDataset(object):
    def __init__(self, target_size=[50,100], shuffle=True, re_generate=False, data_dir='./data/GRID', dst_path='./data/grid_hkl/GRID.h5'):
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
    def next_train_batch(self, batch_size, gen_words=True):
        nb_iterate = len(self.train_paths) // batch_size
        while True:
            if self.shuffle:
                np.random.shuffle(self.train_paths)
            for itr in range(nb_iterate):
                start_pos = itr*batch_size
                yield self.gen_batch(start_pos, batch_size, self.train_paths, gen_words=gen_words)

    """
    generate next validation batch
    """
    def next_val_batch(self, batch_size, test_seen=True, gen_words=False):
        if test_seen:
            paths = self.test_seen_paths
        else:
            paths = self.test_unseen_paths

        nb_iterate = len(paths) // batch_size
        while True:
            for itr in range(nb_iterate):
                start_pos = itr*batch_size
                yield self.gen_batch(start_pos, batch_size, paths, gen_words=gen_words)

    def gen_batch(self, begin, batch_size, paths,gen_words):
        data = np.zeros([batch_size, self.timespecs, self.target_size[0], self.target_size[1], 3])
        label = np.zeros([batch_size, self.max_label_length])
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        source_strs = []

        for i in range(batch_size):
            pos = begin+i
            lip_d, lip_l, lip_label_len, source_str = self.readLipSequences(paths[pos], gen_words=gen_words)
            # print (lip_l, lip_label_len, source_str)
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
    def convertWordToLabel(self, string):
        label = []
        for char in string:
            label.append(ord(char) - ord('a'))
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

    @return
    the returned tuple has four elements, each of which is a list. The first element of the list is the whole sentence, and the following are single words.
    labels: each element is an integer array
    labels_len: each element is the length of the label length
    source_strs: the original word
    frames: the frame begin and end

    """
    def convertAlignToLabels(self, align_file):
        with open(align_file,'r') as f:
            lines = f.readlines()
            words = []
            frames = [[0,self.timespecs-1]]
            sentence_label = []
            source_strs_of_words = []
            source_str = ''
            source_strs = []
            for i,line in enumerate(lines):

                #remove first and last SIL word
                if i ==0:
                    continue
                if i == len(lines)-1:
                    continue

                striped_line = line.rstrip()
                begin,end,word = striped_line.split(' ')
                source_str += word
                words.append(word)
                begin_frame = int(begin) // 1000
                end_frame = int(end) // 1000
                frames.append([begin_frame, end_frame])
                sentence_label.extend(self.convertWordToLabel(word))
                if i!=len(lines)-2:
                    sentence_label.append(26)
                    source_str += ' '

            labels = np.zeros([len(words)+1, self.max_label_length])-1
            labels_len = np.zeros(len(words)+1)
            labels[0, 0:len(sentence_label)] = sentence_label
            labels_len[0] = len(sentence_label)
            source_strs.append(source_str)
            for i,w in enumerate(words):
                labels[i+1,0:len(w)] = self.convertWordToLabel(w)
                labels_len[i+1] = len(w)
                source_strs.append(w)

        return (labels, labels_len, source_strs, frames)

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
    def readLipSequences(self, lipsequence_dir, gen_words):
        total = len(sum([i[2] for i in os.walk(lipsequence_dir)],[]))

        sequence_name = lipsequence_dir.split('/')[-1]
        sequence_owner = lipsequence_dir.split('/')[-2]
        lip_sequence = np.zeros([self.timespecs, self.target_size[0], self.target_size[1], 3])
        if total < self.timespecs-1:
            label = np.zeros(self.max_label_length)-1
            label[0] = self.ctc_blank
            return (lip_sequence, label, 1, "")

        def read_images(frame_intval):
            begin_frame, end_frame = frame_intval
            for i in range(begin_frame, end_frame):
                img_name = '{}/{}.jpg'.format(lipsequence_dir, i)
                lip_sequence[i-begin_frame,...] = self.load_image(img_name, target_size=self.target_size)

        label_path = self.getAlignmentDirOfPerson(sequence_owner, sequence_name)
        labels, labels_len, source_strs, frames = self.convertAlignToLabels(label_path)
        i = 0
        if gen_words and np.random.rand() < 0.5:
            i = np.random.randint(len(frames))
        read_images(frames[i])
        return (lip_sequence, labels[i], labels_len[i],source_strs[i])

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
            #todo
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

