import os
import re
import glob
import logging
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from preprocessing.MyImageDataGenerator import MyImageDataGenerator
from gridBaseDataset import GRIDBaseDataset
from keras.preprocessing.image import * 
from PIL import Image as pil_image

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

class Authenbase40DataGenerator(object):
    """generate data for authentication_database_40"""
    database_pth = os.path.join(CURRENT_PATH, './data/authentication_database_40') 
    def __init__(self, database_pth= database_pth, target_speaker=None, pos_train_number=3):
        super(Authenbase40DataGenerator, self).__init__()
        self.database_pth = database_pth
        self.num_speakers = 40
        self.timespec = 90
        self.img_size = (50, 100)
        self.target_speaker = target_speaker
        self.pos_train_number = pos_train_number

        speakers = np.arange(1, self.num_speakers+1) 
        np.random.seed(10) 
        np.random.shuffle(speakers) 
        self.pretrain_speaker = speakers[0:10] 
        self.pretrain_paths = self.get_sample_paths(self.pretrain_speaker) 
        np.random.seed(10) 
        np.random.shuffle(self.pretrain_paths) 
        self.num_pretrain = len(self.pretrain_paths) 
        split = int(self.num_pretrain*0.8)
        self.pretrain_train_paths = self.pretrain_paths[0:split]
        self.pretrain_val_paths = self.pretrain_paths[split:] 
        self.pretrain_test_paths = self.pretrain_val_paths

        self.augmenter = ImageDataGenerator(
                rotation_range=5, 
                width_shift_range = 0.1,
                height_shift_range=0.1,
                # shear_range=0.05,
                # zoom_range=0.2,
                zoom_range=0.05,
                horizontal_flip=True,
                rescale=1./255
                ) 


        #for speaker authentication
        self.other_speakers = speakers[10:] 
        if target_speaker:
            self.pos_speakers = (target_speaker,)  
            self.other_neg_speakers = [] 
            for x in self.other_speakers:
                if x != target_speaker:
                    self.other_neg_speakers.append(x) 
            self.pos_speaker_paths = self.get_sample_paths(self.pos_speakers)
            self.other_neg_speaker_paths = self.get_sample_paths(self.other_neg_speakers)
            np.random.seed(10) 
            np.random.shuffle(self.pos_speaker_paths)
            np.random.shuffle(self.other_neg_speaker_paths) 
            logging.info("pretrain speakers: {}".format(self.pretrain_speaker) ) 
            logging.info( 'other neg speakers: {}'.format(self.other_neg_speakers) ) 

            self.pos_train_paths = self.pos_speaker_paths[0:self.pos_train_number] 
            self.neg_train_paths = self.pretrain_train_paths
            self.pos_val_paths = self.pos_speaker_paths[self.pos_train_number:self.pos_train_number+3] 
            self.neg_val_paths = self.pretrain_val_paths
            self.pos_test_paths = self.pos_speaker_paths[self.pos_train_number+3:] 
            self.neg_test_paths = self.other_neg_speaker_paths

    def steps_per_epoch(self, batch_size):
        """cal steps per epoch"""
        if self.target_speaker:
            nb_iterate = int(np.ceil(float(len(self.neg_train_paths))/(batch_size // 2) ) ) 
        else:
            nb_iterate = int(np.ceil(float(len(self.pretrain_train_paths)) / batch_size ) ) 
        logging.info( "train_steps is {}".format(nb_iterate) ) 
        return nb_iterate

    def validation_steps(self, batch_size):
        """cal validation steps"""
        if self.target_speaker:
            nb_iterate = int(np.ceil(float(len(self.neg_val_paths))/(batch_size // 2) ) ) 
        else:
            nb_iterate = int(np.ceil(float(len(self.pretrain_val_paths)) / batch_size ) ) 
        logging.info( "validation_steps is {}".format(nb_iterate) ) 
        return nb_iterate

    def test_steps(self, batch_size):
        if self.target_speaker:
            nb_iterate = int(np.ceil(float(len(self.neg_test_paths))/(batch_size // 2) ) ) 
        else:
            nb_iterate = int(np.ceil(float(len(self.pretrain_test_paths)) / batch_size ) ) 
        logging.info( "test_steps is {}".format(nb_iterate) ) 
        return nb_iterate

        
    def next_batch(self, batch_size, phase, shuffle=False):
        """next train for train/val"""
        if phase == 'train':
            if self.target_speaker:
                pos_paths = self.pos_train_paths
                neg_paths = self.neg_train_paths
            else:
                paths = self.pretrain_train_paths
        elif phase == 'val':
            if self.target_speaker:
                pos_paths = self.pos_val_paths
                neg_paths = self.neg_val_paths
            else:
                paths = self.pretrain_val_paths
        elif phase == 'test':
            if self.target_speaker:
                pos_paths = self.pos_test_paths
                neg_paths = self.neg_test_paths
            else:
                paths = self.pretrain_test_paths
        else:
            raise ValueError( 'phase must be one of {train, val, test}') 
        if self.target_speaker:
            nb_iterate = int(np.ceil(float(len(pos_paths))/(batch_size // 2) ) ) 
            neg_path_idx = 0
            neg_path_nums = len(neg_paths) 
        else:
            nb_iterate = int(np.ceil(float(len(paths)) / batch_size ) ) 

        logging.debug( 'iterations per epoch:{}'.format(nb_iterate) ) 
        while True:
            if shuffle:
                if self.target_speaker:
                    np.random.shuffle(pos_paths) 
                    np.random.shuffle(neg_paths) 
                else:
                    np.random.shuffle(paths) 
            start_pos = 0
            for itr in range(nb_iterate):
                current_batch_size = batch_size
                if self.target_speaker:
                    if len(pos_paths)  < start_pos + batch_size//2:
                        current_batch_size = (len(pos_paths)  - start_pos) * 2
                else:
                    if len(paths) < start_pos + batch_size:
                        current_batch_size = len(paths) - start_pos 

                batch_paths = []
                if self.target_speaker:
                    batch_paths.extend(pos_paths[start_pos: start_pos+current_batch_size//2] )
                    if neg_path_idx + current_batch_size // 2 > neg_path_nums:
                        neg_path_idx = 0
                        if shuffle:
                            np.random.shuffle(neg_paths) 
                    batch_paths.extend(neg_paths[neg_path_idx: neg_path_idx+current_batch_size//2] )
                    # batch_paths.extend(np.random.choice(neg_paths, size=current_batch_size//2, replace=False) ) 
                    start_pos += current_batch_size // 2
                    neg_path_idx += current_batch_size // 2 
                else:
                    batch_paths += paths[start_pos: start_pos+current_batch_size] 
                    start_pos += current_batch_size 
                logging.debug("iteration: {}. paths:{}".format(itr, batch_paths) ) 
                yield self.videos_from_frames(batch_paths, phase) 

    def get_sample_paths(self, speakers):
        """get the paths of the speaker"""
        paths = []
        for speaker in speakers:
            paths += glob.glob(os.path.join(self.database_pth, str(speaker), "*") ) 
        return paths


    def speaker_from_path(self, path):
        """get the speaker idx from a path
            the first speaker's index is 0 
        """
        base,video_idx = os.path.split(path) 
        base, speaker = os.path.split(base) 
        return int(speaker)-1
        
    def videos_from_frames(self, paths, phase):
        n = len(paths) 
        lip_shape = (self.timespec, ) + self.img_size  + (3, ) 
        videos = np.zeros((n,)+lip_shape ) 
        num_predict = self.timespec
        if self.target_speaker:
            labels = np.zeros((n, num_predict, 2) ) 
        else:
            labels = np.zeros((n, num_predict, self.num_speakers) ) 

        def load_one_video(path): 
            video = np.zeros(lip_shape) 
            seed = np.random.randint(100000) 
            for i in range(self.timespec):
                frame_path = os.path.join(path, str(i)+ '.bmp') 
                x = img_to_array(load_img(frame_path, target_size=self.img_size) ) 
            # if phase == 'train'or phase == 'val':
                np.random.seed(seed) 
                x = self.augmenter.random_transform(x) 
                x = self.augmenter.standardize(x) 
                video[i] = x
            np.random.seed() 
            return video

        for i,path in enumerate(paths): 
            videos[i,...] = load_one_video(path)  
            current_speaker_idx = self.speaker_from_path(path) 
            if self.target_speaker:
                if current_speaker_idx == self.target_speaker-1:
                    labels[i, :, 1] = 1
                else:
                    labels[i, :, 0] = 1
            else:
                labels[i, :, current_speaker_idx] = 1
        return (videos, labels) 

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)  

    batch_size= 2
    auth_40 = Authenbase40DataGenerator() 
    n = 2
    i=0
    # for videos, labels in auth_40.next_batch(batch_size, phase='train', shuffle = True):
        # print 'auth_40_pretrain train:',labels
        # i +=1
        # if i == n:
            # break
    # i=0
    # for videos, labels in auth_40.next_batch(batch_size, phase='val', shuffle = False):
        # print 'auth_40_pretrain val:',labels
        # i +=1
        # if i == n:
            # break
    auth_40_21 = Authenbase40DataGenerator(target_speaker=24) 
    # i=0
    # for videos, labels in auth_40_21.next_batch(batch_size, phase='train', shuffle = True):
        # print 'auth_40 spaker 21 train:',labels
        # i +=1
        # if i == n:
            # break
    i=0
    for videos, labels in auth_40_21.next_batch(batch_size, phase='val', shuffle = False):
        print 'auth_40_pretrain val:',labels
        i +=1
        if i == n:
            break
