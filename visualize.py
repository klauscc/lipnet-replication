import os
import sys
import logging
import numpy as np
import keras.backend as K

from model.lipnet_res3d import lipnet_res3d
from third_party.deepVizKeras.guided_backprop import GuidedBackprop
from gridSinglePersonAuthentication import GRIDSinglePersonAuthentication

CURRENT_PATH=os.path.dirname(os.path.abspath(__file__) ) 

class VSAVisualizer(object):
    """visualize the vsa network"""
    def __init__(self, lipnet_weight, speaker):
        super(VSAVisualizer, self).__init__()
        K.set_learning_phase(0) 
        pos_n = 25
        self.lipnet_weight = lipnet_weight
        self.speaker = speaker
        self.input_dim = (75, 50, 100, 3) 
        self.max_label_length = 50
        self.data_generator = GRIDSinglePersonAuthentication(auth_person=speaker, pos_n=pos_n)
        self.build_model() 
        
    def build_model(self):
        """build the model"""
        net,test_func = lipnet_res3d(input_dim=self.input_dim, output_dim=self.max_label_length, weights = None, speaker= self.speaker)
        self.test_func = test_func
        self.model = net
        if self.lipnet_weight:
            self.model.load_weights(self.lipnet_weight)
            logging.info( 'loaded weights: {}'.format(self.lipnet_weight) ) 

    def visualize(self, lip_path):
        """visualize the model"""
        mini_batch_lips = self.data_generator.gen_batch(begin=0, 
                batch_size=1,
                paths=(lip_path,) ,
                gen_words=False,
                auth_person=self.speaker,
                scale=1/255.) 
        inputs = mini_batch_lips[0]
        outputs = mini_batch_lips[1] 
        guided_backprop = GuidedBackprop(self.model) 
        mask = guided_bprop.get_mask(inputs)               # compute the gradients
        print mask.shape

def main(path= './data/GRID/lips/s22/prahzs'):
    """main function"""
    lipnet_weight = os.path.join(CURRENT_PATH, './data/checkpoints_grid/grid_vsa_speaker_22.hdf5') 
    vis = VSAVisualizer(lipnet_weight=lipnet_weight, speaker=22) 
    vis.visualize(path) 

if __name__ == '__main__':
    main()
