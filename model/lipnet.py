import os
import numpy as np 
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Activation, Dropout, Conv3D, MaxPooling3D, Flatten,ZeroPadding3D, TimeDistributed, SpatialDropout3D,BatchNormalization,Lambda,GRU,SpatialDropout1D

from core.ctc import ctc_lambda_func

def shared_layers(input, input_dim):
    #STCNN-1
    stcnn1_padding = ZeroPadding3D(padding=(1,2,2), input_shape = input_dim)(input) 
    stcnn1_convolution = Conv3D(32, (3, 5, 5), strides=(1,2,2), kernel_initializer='he_uniform')(stcnn1_padding)
    stcnn1_bn = BatchNormalization()(stcnn1_convolution)
    stcnn1_acti = Activation('relu')(stcnn1_bn)
    #SPATIAL-DROPOUT
    stcnn1_dp = SpatialDropout3D(0.5)(stcnn1_acti)
    #MAXPOOLING-1
    stcnn1_maxpool = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(stcnn1_dp)

    #STCNN-2
    stcnn2_padding = ZeroPadding3D(padding=(1,2,2), input_shape = input_dim)(stcnn1_maxpool)
    stcnn2_convolution = Conv3D(64, (3, 5, 5), strides=(1,1,1), kernel_initializer='he_uniform')(stcnn2_padding)
    # stcnn2_convolution = Conv3D(64, (3, 5, 5), strides=(1,2,2), kernel_initializer='he_uniform')(stcnn2_padding)
    stcnn2_bn = BatchNormalization()(stcnn2_convolution)
    stcnn2_acti = Activation('relu')(stcnn2_bn)
    #SPATIAL-DROPOUT
    stcnn2_dp = SpatialDropout3D(0.5)(stcnn2_acti)
    #MAXPOOLING-2
    stcnn2_maxpool = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(stcnn2_dp)

    #STCNN-3
    stcnn3_padding = ZeroPadding3D(padding=(1,1,1), input_shape = input_dim)(stcnn2_maxpool)
    stcnn3_convolution = Conv3D(96, (3, 3, 3), strides=(1,1,1), kernel_initializer='he_uniform')(stcnn3_padding)
    stcnn3_bn = BatchNormalization()(stcnn3_convolution)
    stcnn3_acti = Activation('relu')(stcnn3_bn)
    #SPATIAL-DROPOUT
    stcnn3_dp = SpatialDropout3D(0.5)(stcnn3_acti)
    #MAXPOOLING-3
    stcnn3_maxpool = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name= 'shared_layers')(stcnn3_dp)
    return stcnn3_maxpool

def authnet(input_dim, output_dim, weights=None):
    input = Input(name='inputs', shape=input_dim)
    feature = shared_layers(input, input_dim) 

    #STCNN-4
    stcnn4_padding = ZeroPadding3D(padding=(0,1,1), input_shape = input_dim)(feature)
    stcnn4_convolution = Conv3D(128, (5, 3, 3), strides=(5,1,1), kernel_initializer='he_uniform')(stcnn4_padding)
    stcnn4_bn = BatchNormalization()(stcnn4_convolution)
    stcnn4_acti = Activation('relu')(stcnn4_bn)
    #SPATIAL-DROPOUT
    stcnn4_dp = SpatialDropout3D(0.5)(stcnn4_acti)
    #MAXPOOLING-3
    stcnn4_maxpool = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 1, 1))(stcnn4_dp)
    auth_flatten = Flatten()(stcnn4_maxpool) 
    auth_out = Dense(34, kernel_initializer= 'he_uniform', activation='softmax', name= 'y_person')(auth_flatten)
    model_auth = Model( inputs=input, outputs=auth_out) 
    model_auth.summary()
    optimizer = Adam(lr=0.0001)
    model_auth.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model_auth

def lipnet_original(input_dim, output_dim, weights=None):
    input = Input(name='inputs', shape=input_dim)
    labels = Input(name='labels', shape=[output_dim], dtype='float32') 
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    stcnn3_maxpool = shared_layers(input, input_dim) 
    stcnn3_maxpool_flatten = TimeDistributed(Flatten())(stcnn3_maxpool, name = 'shared_features_flatten')

    """lipreading layers
    """
    #Bi-GRU-1
    gru_1 = GRU(256, return_sequences=True, name='gru1')(stcnn3_maxpool_flatten)
    gru_1b = GRU(256, return_sequences=True, go_backwards=True, name='gru1_b')(stcnn3_maxpool_flatten)
    gru1_merged = concatenate([gru_1, gru_1b], axis=2)
    #gru1_dropped = SpatialDropout1D(0.5)(gru1_merged)
    gru1_dropped = gru1_merged
    #Bi-GRU-2
    gru_2 = GRU(256, return_sequences=True, name='gru2')(gru1_dropped)
    gru_2b = GRU(256, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_dropped)
    gru2_merged = concatenate([gru_2, gru_2b], axis=2)
    # gru2_dropped = SpatialDropout1D(0.5)(gru2_merged)
    gru2_dropped = gru2_merged
    #fc linear layer
    li = Dense(28, kernel_initializer='he_uniform')(gru2_dropped)
    #ctc loss
    y_pred = TimeDistributed(Activation('softmax', name='y_pred'))(li) 
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out])
    # model = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out])
    if weights and os.path.isfile(weights):
        model.load_weights(weights)

    optimizer = Adam(lr=0.0001)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                        optimizer=optimizer
                        )
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                        # optimizer=optimizer,
                        # metrics= [ 'accuracy'] 
                        # )
    test_func = K.function([input, labels, input_length, label_length, K.learning_phase()], [y_pred, loss_out])
    model.summary()
    return model,test_func

def lipnet(input_dim, output_dim,weights=None):
    input = Input(name='inputs', shape=input_dim)
    labels = Input(name='labels', shape=[output_dim], dtype='float32') 
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    """shared layers
    """
    stcnn3_maxpool = shared_layers(input, input_dim) 
    stcnn3_maxpool_flatten = TimeDistributed(Flatten(), name= 'shared_features_flatten')(stcnn3_maxpool)

    """auth layers
    """
    #STCNN-4
    stcnn4_padding = ZeroPadding3D(padding=(0,1,1), input_shape = input_dim)(stcnn3_maxpool)
    stcnn4_convolution = Conv3D(128, (5, 3, 3), strides=(5,1,1), kernel_initializer='he_uniform')(stcnn4_padding)
    stcnn4_bn = BatchNormalization()(stcnn4_convolution)
    stcnn4_acti = Activation('relu')(stcnn4_bn)
    #SPATIAL-DROPOUT
    stcnn4_dp = SpatialDropout3D(0.5)(stcnn4_acti)
    #MAXPOOLING-3
    stcnn4_maxpool = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 1, 1))(stcnn4_dp)
    auth_flatten = Flatten(name = 'y_person_feature')(stcnn4_maxpool) 
    # auth_dense = Dense(512, kernel_initializer= 'he_uniform')(auth_flatten)
    auth_out = Dense(34, kernel_initializer= 'he_uniform', activation='softmax', name= 'y_person')(auth_flatten)

    """lipreading layers
    """
    #Bi-GRU-1
    gru_1 = GRU(256, return_sequences=True, name='gru1')(stcnn3_maxpool_flatten)
    gru_1b = GRU(256, return_sequences=True, go_backwards=True, name='gru1_b')(stcnn3_maxpool_flatten)
    gru1_merged = concatenate([gru_1, gru_1b], axis=2)
    #gru1_dropped = SpatialDropout1D(0.5)(gru1_merged)
    gru1_dropped = gru1_merged
    #Bi-GRU-2
    gru_2 = GRU(256, return_sequences=True, name='gru2')(gru1_dropped)
    gru_2b = GRU(256, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_dropped)
    gru2_merged = concatenate([gru_2, gru_2b], axis=2)
    # gru2_dropped = SpatialDropout1D(0.5)(gru2_merged)
    gru2_dropped = gru2_merged
    #fc linear layer
    li = Dense(28, kernel_initializer='he_uniform')(gru2_dropped)
    #ctc loss
    y_pred = TimeDistributed(Activation('softmax'), name='y_pred')(li) 
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out, auth_out])
    if weights and os.path.isfile(weights):
        model.load_weights(weights)

    optimizer = Adam(lr=0.0001)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred, 'y_person': 'categorical_crossentropy'},
                        optimizer=optimizer,
                        metrics= [ 'accuracy'] 
                        )
    test_func = K.function([input, labels, input_length, label_length, K.learning_phase()], [y_pred, loss_out])
    return model,test_func
