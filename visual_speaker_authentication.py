import os
import sys
import logging 
import numpy as np
import editdistance 
import pandas as pd
from keras.callbacks import ModelCheckpoint,CSVLogger
import keras.backend as K

from core.ctc import decode_batch 
from common.utils.spell import Spell
from model.lipnet import lipnet 
from model.lipnet_res3d import lipnet_res3d
from gridSinglePersonAuthentication import GRIDSinglePersonAuthentication
from model.stastic_callbacks import StatisticCallback
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'common','dictionaries','grid.txt')

class VisualSpeakerAuthentication(object):
    """train, evaluate, predict visual speaker authentication"""
    # lipnet_weight= './data/checkpoints/lipnet_weights_multiuser_with_auth_2017_07_19__07_24_24-{epoch:02d}-{val_loss:.2f}-{val_acc:.4f}.hdf5'
    # lipnet_resed_weight = './data/checkpoints/lipnet_res3d_weights_multiuser_with_auth_2017_08_07__08_20_49-55-1.55408811569-0.00957627118644.hdf5'
    lipnet_resed_weight = './data/checkpoints_grid/lipnet_res3d_weights_multiuser_with_auth_2017_09_01__04_44_42-44-2.27469587326-0.0199788135593.hdf5'
    def __init__(self, speaker, liveness_net_weight, lipnet_weight=lipnet_resed_weight, pos_n=25):
        super(VisualSpeakerAuthentication, self).__init__()
        self.speaker = speaker
        self.lipnet_weight = lipnet_weight
        self.liveness_net_weight = liveness_net_weight
        self.input_dim = (75, 50, 100, 3) 
        self.max_label_length = 50
        self.data_generator = GRIDSinglePersonAuthentication(auth_person=speaker, pos_n=pos_n)
        self.spell = Spell(path=PREDICT_DICTIONARY)
        self.build_model() 

    def build_model(self):
        """build auth net"""
        # net,test_func = lipnet(input_dim=self.input_dim, output_dim=self.max_label_length, weights = self.lipnet_weight)
        net,test_func = lipnet_res3d(input_dim=self.input_dim, output_dim=self.max_label_length, weights = self.lipnet_weight, speaker= self.speaker)
        self.test_func = test_func
        self.model = net
        # self.model, self.predict_func = build_auth_net(net) 
        # self.model, self.predict_func = build_auth_net_v2(net) 
        if self.liveness_net_weight and os.path.isfile(self.liveness_net_weight):
            self.model.load_weights(self.liveness_net_weight, by_name=True) 
            logging.info( 'loaded weights: {}'.format(self.liveness_net_weight) ) 

    def fit(self, pos_n=25):
        """fit the model"""
        batch_size = 4
        nb_epoch = 30
        auth_person = self.speaker
        # pos_n = 25
        #generators
        steps = 100 * pos_n // 25
        train_gen = self.data_generator.next_batch(batch_size, phase= 'train', shuffle=True)
        val_gen = self.data_generator.next_batch(batch_size, phase= 'test', shuffle=False) 
        #callbacks
        checkpointer = ModelCheckpoint(filepath=self.liveness_net_weight,save_best_only=True,save_weights_only=True, verbose=1, monitor='val_y_person_2_loss')
        # checkpointer = ModelCheckpoint(filepath=self.liveness_net_weight,save_best_only=True,save_weights_only=True, verbose=1, monitor='val_loss')
        log_savepath='./data/logs_grid/visual_speaker_authentication_speaker_{}.csv'.format(self.speaker)
        statisticCallback = StatisticCallback(self.test_func, log_savepath, val_gen, 200)
        logging.info( 'begin training.......\n steps_per_epoch is {}'.format(steps) ) 
        self.model.fit_generator(generator=train_gen, steps_per_epoch=steps,
                    epochs=nb_epoch,initial_epoch=0,verbose=1,
                    callbacks=[checkpointer],
                    validation_data=val_gen, validation_steps=200
                    )

    def predict_on_paths(self, paths, threshold):
        """predict the lips in `paths`
            Args:
            ------
            paths: list. paths of the lip videos.
            threshold: float, 0-1. The threshold to determine if a speaker has spoken the correct sentence. If the edit distance between the predicted of the ground_truth is smaller than threshold, we can say he spoke the correct sentence.

            Returns:
            ------
            all the params are a list of the length of len(paths) 
            speakers.  the speaker, S?. 
            inputs[ 'source_str']. the sentence speaker speaks
            mini_batch_res. the predicted sentence
            mini_batch_auth. 1 if the speaker is the target and 0 if he is an imposter.
            edit_distance. the edit distance between ground_truth sentence and the predicted sentence.
            speaker_accepted. 1 if the mini_batch_auth is 1 and edit_distance is smaller than the threshold
        """
        num_left = len(paths) 
        mini_batch_size = 25
        speakers = []
        source_strs = []
        predicted_strs = []
        speaker_auth = [] 
        edit_distance = []
        speaker_accepted = []
        result = [] 
        idx = 0
        while num_left > 0:
            num_process = min(num_left, mini_batch_size) 
            mini_batch_paths = list(paths[idx:idx+num_process])
            mini_batch_result = self.predict_on_batch_paths(mini_batch_paths, threshold) 
            speakers += mini_batch_result[0] 
            source_strs += mini_batch_result[1] 
            predicted_strs += mini_batch_result[2] 
            speaker_auth += list(mini_batch_result[3][:,1])
            edit_distance += list(mini_batch_result[4])
            speaker_accepted += list(mini_batch_result[5])
            #increase idx
            num_left -= num_process
            idx += num_process
        return { 
                'speaker': speakers,
                'source_str': source_strs,
                'predict_str': predicted_strs,
                'speaker_auth': speaker_auth,
                'edit_distance': edit_distance,
                'speaker_accepted': speaker_accepted
        } 
            
    def predict_on_batch_paths(self, mini_batch_paths, threshold):
        """predict a lip sequnce"""
        mini_batch_size = len(mini_batch_paths) 
        mini_batch_lips = self.data_generator.gen_batch(begin=0, 
                batch_size=mini_batch_size,
                paths=mini_batch_paths,
                gen_words=False,
                auth_person=self.speaker,
                scale=1/255.) 
        inputs = mini_batch_lips[0]
        outputs = mini_batch_lips[1] 
        input_list = [inputs[ 'inputs'], inputs[ 'labels'],
                inputs[ 'input_length'], inputs[ 'label_length']  ]
        mini_batch_res, mini_batch_auth, speaker_accepted, edit_distance = self.predict_on_batch(inputs, threshold) 
        mini_batch_speakers = [self.data_generator.get_speaker_idx_of_path(path)+1 for path in mini_batch_paths ]
        return mini_batch_speakers, inputs[ 'source_str'], mini_batch_res, mini_batch_auth, edit_distance, speaker_accepted

    def predict_on_batch(self, inputs_dict, threshold):
        """predict on the inputs.
            Args:
            --------
            inputs_dict: a dict. keys:
                all the parameters has the same length.
                inputs: a 4-rank array: (batch_size, 75, 50, 100, 3). a batch of lip sequences
                labels: a 2-rank array. the label vector of each lip sequence
                input_length: a 1-rank array. the length of each input
                label_length: a 1-rank array. the label length of each input
                threshold: float, 0-1. The threshold to determine if a speaker has spoken the correct sentence. If the edit distance between the predicted of the ground_truth is smaller than threshold, we can say he spoke the correct sentence.
            
            Return:
            ---------
            mini_batch_res: a 1-rank array. each value is the predicted string of the inputs
            mini_batch_auth: 1-rank array. Wether the speaker is the target speaker
            speaker_accepted: Boolen. Wether the speaker is accepted. A speaker is accepted only if he is the target speaker and he speaks the correct sentence.
        """
        input_list = [inputs_dict[ 'inputs'], inputs_dict[ 'labels'],
                inputs_dict[ 'input_length'], inputs_dict[ 'label_length']  ]
        mini_batch_res,loss, mini_batch_auth = decode_batch(self.test_func, input_list) 
        mini_batch_auth = np.mean(mini_batch_auth ,axis=1) 
        mini_batch_size = len(inputs_dict[ 'inputs'] ) 
        speaker_accepted = np.zeros([mini_batch_size] ) 
        edit_distances = np.zeros([mini_batch_size] ) 
        for i in range(mini_batch_size):
            ground_truth = inputs_dict[ 'source_str'][i] 
            mini_batch_res[i] = self.spell.sentence(mini_batch_res[i] )
            edit_distance = editdistance.eval(mini_batch_res[i], ground_truth) / float(inputs_dict[ 'label_length'][i]  ) 
            edit_distances[i] = edit_distance 
            if mini_batch_auth[i,1] >= 0.84:  # speaker is the target
                if edit_distance < threshold: # speaker speak correct password
                    speaker_accepted[i] = 1 
                logging.debug( 'target speaker. predicted: {} | ground_truth: {}'.format(mini_batch_res[i], ground_truth) ) 
            else:
                logging.debug( 'imposter. predicted: {} | ground_truth: {}'.format(mini_batch_res[i], ground_truth) ) 
        return mini_batch_res, mini_batch_auth, speaker_accepted, edit_distances

def predict_all(speaker, idx, threshold=0.3):
    """predict all the videos and save the result"""
    
    target_csv_save_name = os.path.join(CURRENT_PATH, './data/vsa_result/S{}_target_{}.csv'.format(speaker,idx) ) 
    imposter_csv_save_name = os.path.join(CURRENT_PATH, './data/vsa_result/S{}_imposter_{}.csv'.format(speaker, idx) ) 

    liveness_net_weight = './data/checkpoints/grid_vsa_speaker_{}.hdf5'.format(speaker) 
    vsa = VisualSpeakerAuthentication(speaker=speaker, liveness_net_weight=liveness_net_weight, lipnet_weight=None) 
    target_video_paths = vsa.data_generator.pos_test_paths
    imposter_video_paths = vsa.data_generator.neg_test_paths
    target_result = vsa.predict_on_paths(target_video_paths, threshold) 
    target_df = pd.DataFrame(data=target_result) 
    target_df.to_csv(target_csv_save_name, float_format= '%.3f', index_label= 'index') 
    imposter_result = vsa.predict_on_paths(imposter_video_paths, threshold) 
    imposter_df = pd.DataFrame(data=imposter_result) 
    imposter_df.to_csv(imposter_csv_save_name, float_format= '%.3f', index_label= 'index') 

def eval(speaker=25, phase= 'val'):
    batch_size=8
    steps = 225
    threshold = 0.97
    liveness_net_weight = './data/checkpoints_grid/grid_vsa_speaker_{}.hdf5'.format(speaker) 
    visual_speaker_authentication = VisualSpeakerAuthentication(speaker=speaker, liveness_net_weight=liveness_net_weight, lipnet_weight=None) 
    # visual_speaker_authentication = VisualSpeakerAuthentication(speaker=speaker, liveness_net_weight=liveness_net_weight) 
    val_gen = visual_speaker_authentication.data_generator.next_batch(batch_size, phase= phase, shuffle=False) 
    y_pred = []
    y_true = []
    y_prob = []
    for i, (inputs, outputs) in enumerate(val_gen):
        # mini_batch_res,loss, mini_batch_auth = decode_batch(visual_speaker_authentication.test_func, input_list) 
        mini_batch_res, mini_batch_auth, speaker_accepted,edit_distance = visual_speaker_authentication.predict_on_batch(inputs, 0.3) 
        # mini_batch_auth = np.mean(mini_batch_auth ,axis=1) 
        y_true += list(np.argmax(outputs[ 'y_person_2'][:,0,:] ,axis=1))
        y_prob += list(mini_batch_auth)
        y_pred += list(mini_batch_auth[:,1] > threshold ) 
        # print mini_batch_auth, np.argmax(outputs[ 'y_person_2'][:,0,:] ,axis=1)
        if i == steps-1:
            break
    y_prob = np.array(y_prob)
    report = classification_report(y_true, y_pred, digits=4)
    softmax_save_path = './data/logs_grid/grid_vsa_speaker_{}_softmax_{}.csv'.format(visual_speaker_authentication.speaker, phase) 
    np.savetxt(softmax_save_path, np.c_[y_prob, y_true], fmt= "%.5f", delimiter= ',')
    fpr, tpr, threshold = roc_curve(y_true, y_prob[:,1], drop_intermediate=False)
    tpr_fpr_save_path = './data/logs_grid/grid_vsa_speaker_{}_tpr_fpr_{}.csv'.format(visual_speaker_authentication.speaker, phase) 
    np.savetxt(tpr_fpr_save_path, np.c_[fpr, tpr, threshold], fmt="%.5f", delimiter=',', header='fpr,tpr,threshold')
    print report

def train(speaker=25, pos_n=25):
    liveness_net_weight = './data/checkpoints_grid/grid_vsa_speaker_{}.hdf5'.format(speaker) 
    visual_speaker_authentication = VisualSpeakerAuthentication(speaker=speaker, liveness_net_weight=liveness_net_weight, pos_n=pos_n) 
    visual_speaker_authentication.fit(pos_n = pos_n) 

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
	    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
	    datefmt='%a, %d %b %Y %H:%M:%S'
	    )
    import sys
    speaker = int(sys.argv[2] ) 
    if sys.argv[1] == 'train':
        pos_n = int(sys.argv[3])
        train(speaker, pos_n)
    elif sys.argv[1] == 'eval':
        K.set_learning_phase(0) 
        phase = sys.argv[3] 
        eval(speaker, phase)
    elif sys.argv[1] == 'test':
        K.set_learning_phase(0) 
        idx = sys.argv[3] 
        print( 'test {}'.format(idx) ) 
        predict_all(speaker=speaker, idx=idx) 
