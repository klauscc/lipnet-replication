import os
import sys
import logging 
import numpy as np
import editdistance 
from keras.callbacks import ModelCheckpoint,CSVLogger
import keras.backend as K

from core.ctc import decode_batch 
from common.utils.spell import Spell
from model.lipnet import lipnet 
from model.lipnet_res3d import lipnet_res3d
from model.auth_net import build_auth_net, build_auth_net_v2
from gridSinglePersonAuthentication import GRIDSinglePersonAuthentication
from model.stastic_callbacks import StatisticCallback
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'common','dictionaries','grid.txt')

class VisualSpeakerAuthentication(object):
    """train, evaluate, predict visual speaker authentication"""
    lipnet_weight= './data/checkpoints/lipnet_weights_multiuser_with_auth_2017_07_19__07_24_24-{epoch:02d}-{val_loss:.2f}-{val_acc:.4f}.hdf5'
    lipnet_resed_weight = './data/checkpoints/lipnet_res3d_weights_multiuser_with_auth_2017_07_25__09_22_08-52-10.9587259293-0.0775..hdf5'
    def __init__(self, speaker, liveness_net_weight, lipnet_weight=lipnet_resed_weight):
        super(VisualSpeakerAuthentication, self).__init__()
        self.speaker = speaker
        self.lipnet_weight = lipnet_weight
        self.liveness_net_weight = liveness_net_weight
        self.input_dim = (75, 50, 100, 3) 
        self.max_label_length = 50
        self.grid = GRIDSinglePersonAuthentication(auth_person=speaker)
        self.spell = Spell(path=PREDICT_DICTIONARY)
        self.build_model() 

    def build_model(self):
        """build auth net"""
        # net,test_func = lipnet(input_dim=self.input_dim, output_dim=self.max_label_length, weights = self.lipnet_weight)
        net,test_func = lipnet_res3d(input_dim=self.input_dim, output_dim=self.max_label_length, weights = self.lipnet_weight)
        self.test_func = test_func
        self.model, self.predict_func = build_auth_net(net) 
        # self.model, self.predict_func = build_auth_net_v2(net) 
        if self.liveness_net_weight and os.path.isfile(self.liveness_net_weight):
            self.model.load_weights(self.liveness_net_weight) 

    def fit(self):
        """fit the model"""
        batch_size = 2
        nb_epoch = 10
        auth_person = self.speaker
        pos_n = 25
        #generators
        train_gen = self.grid.next_batch(batch_size, phase= 'train', shuffle=True)
        val_gen = self.grid.next_batch(batch_size, phase= 'val', shuffle=False) 
        #callbacks
        checkpointer = ModelCheckpoint(filepath=self.liveness_net_weight,save_best_only=False,save_weights_only=True)
        log_savepath='./data/logs/visual_speaker_authentication_speaker_{}.csv'.format(self.speaker)
        statisticCallback = StatisticCallback(self.test_func, log_savepath, val_gen, 200)
        self.model.fit_generator(generator=train_gen, steps_per_epoch=pos_n *2// batch_size,
                    nb_epoch=nb_epoch,initial_epoch=0,
                    callbacks=[checkpointer],
                    # validation_data=val_gen, validation_steps= 100
                    )

    def predict_on_batch(self, mini_batch_paths, threshold=0.1):
        """predict a lip sequnce"""
        mini_batch_size = len(mini_batch_paths) 
        mini_batch_lips = self.grid.gen_batch(begin=0, 
                batch_size=mini_batch_size,
                paths=mini_batch_paths,
                gen_words=False,
                auth_person=self.speaker,
                scale=1/255.) 
        inputs = mini_batch_lips[0]
        input_list = [inputs[ 'inputs'], inputs[ 'labels'],
                inputs[ 'input_length'], inputs[ 'label_length']  ]
        mini_batch_res, mini_batch_auth = decode_batch(self.test_func, input_list) 
        speaker_accepted = np.zeros([mini_batch_size] ) 
        for i in range(mini_batch_size):
            if np.argmax(mini_batch_auth[i]) == 1:  # speaker is the target
                mini_batch_res[i] = self.spell.sentence(mini_batch_res[i] )
                ground_truth = inputs[ 'source_str'][i] 
                edit_distance = editdistance.eval(mini_batch_res[i], ground_truth) 
                if edit_distance < threshold: # speaker speak correct password
                    speaker_accepted[i] = 1 
        return speaker_accepted

def eval():
    speaker = 24 
    liveness_net_weight = './data/checkpoints/visual_speaker_authentication_speaker_{}.hdf5'.format(speaker) 
    batch_size=8
    steps = 50
    threshold = 0.8
    visual_speaker_authentication = VisualSpeakerAuthentication(speaker=speaker, liveness_net_weight=liveness_net_weight, lipnet_weight=None) 
    val_gen = visual_speaker_authentication.grid.next_batch(batch_size, phase= 'val', shuffle=False) 
    y_pred = []
    y_true = []
    y_prob = []
    for i, (inputs, outputs) in enumerate(val_gen):
        input_list = [inputs[ 'inputs'], inputs[ 'labels'],
                inputs[ 'input_length'], inputs[ 'label_length']  ]
        mini_batch_res, mini_batch_auth = decode_batch(visual_speaker_authentication.predict_func, input_list) 
        y_true += list(np.argmax(outputs[ 'y_person_auth'],axis=1))
        y_prob += list(mini_batch_auth)
        y_pred += list(mini_batch_auth[:,1] > threshold ) 
        print mini_batch_auth, outputs[ 'y_person_auth'] 
        if i == steps:
            break
    y_prob = np.array(y_prob)
    report = classification_report(y_true, y_pred, digits=4)
    softmax_save_path = './data/logs/visual_speaker_authentication_speaker_{}_softmax.csv'.format(visual_speaker_authentication.speaker) 
    np.savetxt(softmax_save_path, np.c_[y_prob, y_true], fmt= "%.5f", delimiter= ',')
    fpr, tpr, threshold = roc_curve(y_true, y_prob[:,1] )
    tpr_fpr_save_path = './data/logs/visual_speaker_authentication_speaker_{}_tpr_fpr.csv'.format(visual_speaker_authentication.speaker) 
    np.savetxt(tpr_fpr_save_path, np.c_[fpr, tpr, threshold], fmt="%.5f", delimiter=',', header='fpr,tpr,threshold')





def main():
    speaker = 24 
    liveness_net_weight = './data/checkpoints/visual_speaker_authentication_speaker_{}.hdf5'.format(speaker) 
    visual_speaker_authentication = VisualSpeakerAuthentication(speaker=speaker, liveness_net_weight=liveness_net_weight) 
    visual_speaker_authentication.fit() 

if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'train':
        main()
    else:
        K.set_learning_phase(0) 
        eval()
