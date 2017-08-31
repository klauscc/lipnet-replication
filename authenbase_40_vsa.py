from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
import os
import sys
import logging 
import numpy as np
import editdistance 
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping
import keras.backend as K 
from core.ctc import decode_batch 
from common.utils.spell import Spell
from model.lipnet import lipnet 
from model.lipnet_res3d import lipnet_res3d,auth_net_res
from model.auth_net import build_auth_net, build_auth_net_v3, build_auth_net_v4 , build_auth_net_res_v2
from model.stastic_callbacks import StatisticCallback 
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report

from authenbase_40_data_generator import Authenbase40DataGenerator

class AuthenBase40VSA(object): 
    """train, evaluate, predict visual speaker authentication""" 
    def __init__(self, data_generator, pretrain_weight=None, auth_weight=None, speaker=None, net='res'):
        super(AuthenBase40VSA, self).__init__()
        self.speaker = speaker
        self.pretrain_weight = pretrain_weight
        self.auth_weight = auth_weight
        self.input_dim = (90, 50, 100, 3) 
        self.data_generator = data_generator
        if self.speaker:
            self.output_dim=2
        else:
            self.output_dim=40
        self.net = net
        self.build_model() 

    def build_model(self):
        """build auth net"""
        if self.net is 'res':
            self.model = auth_net_res(self.input_dim, self.output_dim) 
        elif self.net is 'v4':
            self.model = build_auth_net_v4(self.input_dim, self.output_dim) 
        elif self.net is 'res_v2':
            self.model = build_auth_net_res_v2(self.input_dim, self.output_dim) 
        else:
            raise(ValueError( 'net must be one of {res, v4, res_v2}') ) 
        # self.model = build_auth_net_v3(self.input_dim, self.output_dim) 
        self.model.summary() 
        plot_model(self.model, to_file=self.net+'.png', show_shapes=True, show_layer_names=False) 
        if self.auth_weight and os.path.isfile(self.auth_weight) :
            logging.info( 'loading auth_weight: {}'.format(self.auth_weight) ) 
            self.model.load_weights(self.auth_weight, by_name=True)
        elif self.pretrain_weight and os.path.isfile(self.pretrain_weight):
            logging.info( 'loading pretrain_weight: {}'.format(self.pretrain_weight) ) 
            self.model.load_weights(self.pretrain_weight, by_name=True)
        logging.info( 'model loaded successful...') 

    def fit(self, validation_steps=None):
        """fit the model"""
        batch_size = 4
        nb_epoch = 40
        # validation_steps=290
        #generators
        train_gen = self.data_generator.next_batch(batch_size, phase= 'train', shuffle=True)
        val_gen = self.data_generator.next_batch(batch_size, phase= 'val', shuffle=False) 
        #callbacks
        if self.speaker:
            model_save_path = self.auth_weight
        else:
            model_save_path = self.pretrain_weight
        checkpointer = ModelCheckpoint(filepath=model_save_path, save_best_only=True,save_weights_only=True, verbose=1)
        early_stop = EarlyStopping(monitor= 'val_loss', patience=10) 

        if validation_steps is None:
            validation_steps = self.data_generator.validation_steps(batch_size)*4

        self.model.fit_generator(generator=train_gen, steps_per_epoch=self.data_generator.steps_per_epoch(batch_size) ,
                    epochs=nb_epoch,initial_epoch=0,
                    # callbacks=[checkpointer, early_stop],
                    callbacks=[checkpointer],
                    validation_data=val_gen, validation_steps=validation_steps 
                    )

        def evaluate(self, steps=50, threshold=0.8):
            batch_size = 2
            val_gen = v

def evaluate(vsa, gen, steps, phase, threshold):
    y_pred = []
    y_true = []
    y_prob = []
    for i, (inputs, outputs) in enumerate(gen):
        mini_batch_auth_time = vsa.model.predict(inputs) 
        # mini_batch_auth_time_pred = np.array(mini_batch_auth_time > [1-threshold, threshold], dtype=float)
        # mini_batch_auth = np.mean(mini_batch_auth_time_pred, axis=1) 
        mini_batch_auth = np.mean(mini_batch_auth_time, axis=1) 
        y_true += list(np.argmax(outputs[:,0,:] ,axis=1))
        y_prob += list(mini_batch_auth)
        y_pred += list(mini_batch_auth[:,1] > threshold ) 
        # if not np.all(list(mini_batch_auth[:,1] > 0.95 ) == list(np.argmax(outputs[:,0,:] ,axis=1))):
            # print mini_batch_auth_time
        # print mini_batch_auth, outputs[ 'y_person_auth'] 
        if i == steps-1:
            break
    y_prob = np.array(y_prob)
    report = classification_report(y_true, y_pred, digits=4)
    logging.info( 'phase: {} report: {}'.format(phase, report) ) 
    report_save_path = './data/logs/visual_speaker_authentication_speaker_{}_{}_report_{}.csv'.format(net, speaker, phase) 
    with open(report_save_path, 'w+') as f:
        f.write(report) 
    softmax_save_path = './data/logs/visual_speaker_authentication_speaker_{}_{}_softmax_{}.csv'.format(net, speaker, phase) 
    np.savetxt(softmax_save_path, np.c_[y_prob, y_true], fmt= "%.5f", delimiter= ',')
    fpr, tpr, threshold = roc_curve(y_true, y_prob[:,1], drop_intermediate=False)
    tpr_fpr_save_path = './data/logs/visual_speaker_authentication_speaker_{}_{}_tpr_fpr_{}.csv'.format(net, speaker, phase) 
    np.savetxt(tpr_fpr_save_path, np.c_[fpr, tpr, threshold], fmt="%.5f", delimiter=',', header='fpr,tpr,threshold')
    #cal eer and eer_threshold
    eer_index = np.argmin(np.abs(tpr+fpr-1) ) 
    eer_threshold = threshold[eer_index] -0.02
    eer = np.abs(tpr[eer_index] +fpr[eer_index] -1)/2 
    return eer, eer_threshold

def eval(speaker, weight, net, threshold=0.5):
    auth_weight = weight
    batch_size=2
    data_generator = Authenbase40DataGenerator(target_speaker=speaker) 
    vsa = AuthenBase40VSA(data_generator=data_generator, speaker=speaker, auth_weight=auth_weight, pretrain_weight=None, net=net) 
    val_gen = data_generator.next_batch(batch_size, phase= 'val', shuffle=False) 
    test_gen = data_generator.next_batch(batch_size, phase= 'test', shuffle=False) 
    val_steps = data_generator.validation_steps(batch_size)*4
    test_steps = data_generator.test_steps(batch_size) 
    eer,eer_threshold = evaluate(vsa, val_gen, val_steps, 'val',threshold) 
    logging.info( 'val eer: {}, threshold: {}'.format(eer, eer_threshold) )
    evaluate(vsa, test_gen, test_steps, 'test', eer_threshold) 


def finetune(speaker, net, auth_weight, pretrain_weight): 
    speaker = speaker
    auth_weight = auth_weight
    pretrain_weight = pretrain_weight
    data_generator = Authenbase40DataGenerator(target_speaker=speaker, pos_train_number=3) 
    vsa = AuthenBase40VSA(data_generator=data_generator, speaker=speaker, pretrain_weight = pretrain_weight, auth_weight=auth_weight, net=net) 
    vsa.fit() 

def train(net, pretrain_weight):
    speaker = None
    auth_weight = None
    pretrain_weight = pretrain_weight
    data_generator = Authenbase40DataGenerator(target_speaker=speaker) 
    vsa = AuthenBase40VSA(data_generator=data_generator, speaker=speaker, pretrain_weight = pretrain_weight, auth_weight=auth_weight, net=net) 
    vsa.fit() 

if __name__ == '__main__':
    import sys
    logging.getLogger().setLevel(logging.INFO)  
    net = 'res_v2'
    pretrain_weight = './data/checkpoints/authenbase40_vsa_{}_pretrained.hdf5'.format(net) 
    
    if sys.argv[1] == 'train':
        train(net, pretrain_weight)
    elif sys.argv[1] == 'finetune':
        speaker = int(sys.argv[2])
        auth_weight = './data/checkpoints/authenbase40_vsa_{}_speaker_{}.hdf5'.format(net, speaker) 
        finetune(speaker, net, auth_weight, pretrain_weight) 
        K.clear_session() 
    else:
        K.set_learning_phase(0) 
        speaker = int(sys.argv[2])
        auth_weight = './data/checkpoints/authenbase40_vsa_{}_speaker_{}.hdf5'.format(net, speaker) 
        eval(speaker, auth_weight, net)
