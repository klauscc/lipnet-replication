import os
import numpy as np
import hickle as hkl
import keras.backend as K
from PIL import Image as pil_image

class vsrDatasetLip():

    def __init__(self, target_size=[50,100], re_generate=False ,data_dir='./data/vsr', hkl_path='./data/hkl/vsr.hkl'):
        self.data_root = data_dir
        self.hkl_path = hkl_path
        self.re_generate = re_generate
        
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.target_size = target_size


    """
    get image data from data dir
    """
    def get_data_from_dir(self):
        data_path = self.data_root

        channel = 3
        
        persons = 7
        words = 10
        sample_train = 20
        sample_val = 15
        timespecs = 30

        train_n = (persons-1)*words*sample_train
        val_n = words*sample_val

        x_train = np.zeros([train_n,timespecs, self.target_size[0], self.target_size[1],channel])
        y_train = np.zeros([train_n])

        x_val = np.zeros([val_n,timespecs, self.target_size[0], self.target_size[1],channel])
        y_val = np.zeros([val_n])

        i=0
        for person in xrange(1,persons):
            for word in xrange(1, words+1):
                word_path = os.path.join(data_path,str(person),str(word))
                sample_paths = os.listdir(word_path)
                for sample in sample_paths:
                    image_path = os.path.join(word_path,sample)
                    for t in range(30):
                        path = image_path+'/'+str(t)+'.bmp'
                        print 'train_set,loading',path
                        img = self.load_image(path ,target_size=self.target_size)
                        img = np.asarray(img,dtype=K.floatx())
                        img = self.preprocess_input(img)
                        x_train[i,t,:,:] = img
                        y_train[i] = word-1 
                    i = i+1

        person = 7
        i=0
        for word in xrange(1, words+1):
            word_path = os.path.join(data_path,str(person),str(word))
            sample_paths = os.listdir(word_path)
            for sample in sample_paths:
                image_path = os.path.join(word_path,sample)
                for t in range(30):
                    path = image_path+'/'+str(t)+'.bmp'
                    print 'val_set, loading',path
                    img = self.load_image(path ,target_size=self.target_size)
                    img = np.asarray(img,dtype=float)
                    img = self.preprocess_input(img)
                    x_val[i,t,:,:] = img
                    y_val[i] = word-1 
                i=i+1

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        return [x_train,y_train,x_val,y_val]

                



    def load_data(self):
        hkl_path = self.hkl_path
        if os.path.isfile(hkl_path) and not self.re_generate:
            return hkl.load(hkl_path)
        else:
            z = self.get_data_from_dir()
            hkl_dir = os.path.dirname(hkl_path)
            os.system('mkdir -p ' + hkl_dir)
            hkl.dump(z, hkl_path)
            return z

    def load_image(self,path, grayscale=False, target_size=None):
        img = pil_image.open(path)
        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        
        if target_size:
            img = img.resize((target_size[1],target_size[0]))
        return img

    def preprocess_input(self,x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

        
if __name__ == "__main__":
    dataset = DatasetLip()
    dataset.get_data_from_dir()
