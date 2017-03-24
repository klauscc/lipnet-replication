from keras.models import Sequential,Model
from keras.layers import Input, Dense, Activation, Dropout, Convolution3D, MaxPooling3D, Flatten,ZeroPadding3D, TimeDistributed, SpatialDropout3D,BatchNormalization,Lambda,GRU,merge
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from keras.models import load_model


def lipnet(input_dim, output_dim,weights=None):
    input = Input(name='inputs', shape=input_dim)
    labels = Input(name='labels', shape=[output_dim], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    #STCNN-1
    stcnn1_padding = ZeroPadding3D(padding=(1,2,2), input_shape = input_dim)(input)
    stcnn1_convolution = Convolution3D(32, 3, 5, 5, subsample=(1,2,2))(stcnn1_padding)
    stcnn1_bn = BatchNormalization()(stcnn1_convolution)
    stcnn1_acti = Activation('relu')(stcnn1_bn)
    #SPATIAL-DROPOUT
    stcnn1_dp = SpatialDropout3D(0.5)(stcnn1_acti)
    #MAXPOOLING-1
    stcnn1_maxpool = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(stcnn1_dp)

    #STCNN-2
    stcnn2_padding = ZeroPadding3D(padding=(1,2,2), input_shape = input_dim)(stcnn1_maxpool)
    stcnn2_convolution = Convolution3D(64, 3, 5, 5, subsample=(1,2,2))(stcnn2_padding)
    stcnn2_bn = BatchNormalization()(stcnn2_convolution)
    stcnn2_acti = Activation('relu')(stcnn2_bn)
    #SPATIAL-DROPOUT
    stcnn2_dp = SpatialDropout3D(0.5)(stcnn2_acti)
    #MAXPOOLING-2
    stcnn2_maxpool = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(stcnn2_dp)

    #STCNN-3
    stcnn3_padding = ZeroPadding3D(padding=(1,2,2), input_shape = input_dim)(stcnn2_maxpool)
    stcnn3_convolution = Convolution3D(64, 3, 3, 3, subsample=(1,2,2))(stcnn2_padding)
    stcnn3_bn = BatchNormalization()(stcnn2_convolution)
    stcnn3_acti = Activation('relu')(stcnn2_bn)
    #SPATIAL-DROPOUT
    stcnn3_dp = SpatialDropout3D(0.5)(stcnn2_acti)
    #MAXPOOLING-3
    stcnn3_maxpool = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(stcnn2_dp)

    stcnn3_maxpool_flatten = TimeDistributed(Flatten())(stcnn3_maxpool)

    #Bi-GRU-1
    gru_1 = GRU(256, return_sequences=True, name='gru1')(stcnn3_maxpool_flatten)
    gru_1b = GRU(256, return_sequences=True, go_backwards=True, name='gru1_b')(stcnn3_maxpool_flatten)
    gru1_merged = merge([gru_1, gru_1b], mode='concat', concat_axis=2)
    #Bi-GRU-2
    gru_2 = GRU(256, return_sequences=True, name='gru2')(gru1_merged)
    gru_2b = GRU(256, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_merged)
    gru2_merged = merge([gru_2, gru_2b], mode='concat', concat_axis=2)
    #fc linear layer
    li = Dense(28)(gru2_merged)
    #ctc loss
    y_pred = TimeDistributed(Activation('softmax', name='y_pred'))(li) 
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(input=[input, labels, input_length, label_length], output=[loss_out])
    if weights and os.path.isfile(weights):
        model.load_weights(weights)

    optimizer = Adam(lr=0.0001)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                        optimizer=optimizer
                        )
    model.summary()
    return model

def ctc_lambda_func(args):
     y_pred, labels, input_length, label_length = args

     # the 2 is critical here since the first couple outputs of the RNN
     # tend to be garbage:
     y_pred = y_pred[:, 2:, :]
     return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
