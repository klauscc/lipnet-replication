from model.resnet3d import basic_block, _conv_bn_relu3D, _bn_relu,bottleneck
from keras.layers import Conv3D, Input, AveragePooling3D, Flatten, SpatialDropout3D, Dense, GRU, TimeDistributed, concatenate, Activation, Lambda, Conv2D,AveragePooling2D,Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Model
import keras.backend as K
import os
from model.lipnet import ctc_lambda_func
import logging

def shared_layers(input_tensor):

    kernel_regularizer = l2(1e-4) 

    conv1 = Conv3D(32, (3,5,5), strides=(1,2,2), padding= 'same', kernel_initializer="he_normal", kernel_regularizer=kernel_regularizer)(input_tensor) 
    conv2 = basic_block(32)(conv1)

    conv3 = Conv3D(64, (3,3,3), strides=(1,2,2), padding= 'same', kernel_initializer="he_normal", kernel_regularizer=kernel_regularizer)(conv2) 
    conv4 = basic_block(64)(conv3)  

    conv5 = Conv3D(96, (3,3,3), strides=(1,2,2), padding= 'same', kernel_initializer="he_normal", kernel_regularizer=kernel_regularizer)(conv4) 
    conv6 = basic_block(96)(conv5)  
    return conv6

def auth_net(input_tensor, speaker):
    if not speaker:
        output_dim = 34
    else:
        output_dim = 2
    kernel_regularizer = l2(1e-4) 
    conv7 = TimeDistributed(Conv2D(128, (3,3), strides=(2,2), padding= 'same', kernel_initializer= 'he_normal', kernel_regularizer=kernel_regularizer  ) )(input_tensor)
    conv8 = basic_block(256)(conv7)  
    block_out = _bn_relu(conv8) 
    average_pooling = TimeDistributed(AveragePooling2D(pool_size= (2,3) ) )(block_out) #90x256
    average_pooling = TimeDistributed(Flatten())(average_pooling) 
    average_pooling = TimeDistributed(Dropout(0.5) )(average_pooling) 
    hidden_1 = TimeDistributed(Dense(512, kernel_initializer= 'he_normal', activation= 'relu', kernel_regularizer=kernel_regularizer) ) (average_pooling) 
    hidden_1 = TimeDistributed(Dropout(0.5))(hidden_1) 
    auth_out = TimeDistributed(Dense(output_dim, kernel_initializer= 'he_normal', activation= 'softmax', kernel_regularizer=kernel_regularizer  ),name= 'y_person_'+str(output_dim) )(hidden_1) #90xoutput_dim
    return auth_out

def auth_net_res(input_dim, output_dim):
    input = Input(shape=input_dim, name= 'inputs') 
    feature = shared_layers(input) 
    person_auth = auth_net(feature, output_dim) 
    model_auth = Model(inputs=[input], outputs=[person_auth]) 
    optimizer = Adam(lr=0.0001)
    model_auth.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model_auth 

def liveness_net(input_tensor,labels, input_length, label_length ):
    block_out = SpatialDropout3D(0.5)(input_tensor)  
    pool1 = Conv3D(96, (1,2,2), strides=(1,2,2), kernel_initializer= 'he_normal', padding= 'same', kernel_regularizer=l2(1e-4) )(block_out)
    flatten1 = TimeDistributed( Flatten())(pool1)  

    #Bi-GRU-1
    gru_1 = GRU(256, return_sequences=True, name='gru1')(flatten1)
    gru_1b = GRU(256, return_sequences=True, go_backwards=True, name='gru1_b')(flatten1)
    gru1_merged = concatenate([gru_1, gru_1b], axis=2)
    # gru1_dropped = SpatialDropout1D(0.5)(gru1_merged)
    gru1_dropped = gru1_merged
    #Bi-GRU-2
    gru_2 = GRU(256, return_sequences=True, name='gru2')(gru1_dropped)
    gru_2b = GRU(256, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_dropped)
    gru2_merged = concatenate([gru_2, gru_2b], axis=2)
    # gru2_dropped = SpatialDropout1D(0.5)(gru2_merged)
    gru2_dropped = gru2_merged
    #fc linear layer
    li = Dense(28, kernel_initializer='he_normal')(gru2_dropped)
    #ctc loss
    y_pred = TimeDistributed(Activation('softmax'), name='y_pred')(li) 

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    return y_pred, loss_out

def compile_model(model, speaker):
    if not speaker:
        output_dim = 34
    else:
        output_dim = 2
    optimizer = Adam(lr=0.0001)
    if speaker:
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred, 'y_person_'+str(output_dim) : 'categorical_crossentropy'},
                loss_weights = { 'ctc':0.5, 'y_person_'+str(output_dim): 1.  },
                            optimizer=optimizer,
                            metrics= [ 'accuracy'] 
                            )
    else:
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred, 'y_person_'+str(output_dim) : 'categorical_crossentropy'},
                loss_weights = { 'ctc':1., 'y_person_'+str(output_dim): 0.5 },
                            optimizer=optimizer,
                            metrics= [ 'accuracy'] 
                            )

def lipnet_res3d(input_dim, output_dim, weights, speaker=None):
    input = Input(name= 'inputs', shape=input_dim) 
    labels = Input(name='labels', shape=[output_dim], dtype='float32') 
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    feature = shared_layers(input) 
    person_auth = auth_net(feature, speaker) 
    y_pred, ctc_loss = liveness_net(feature, labels, input_length, label_length) 
    model = Model(inputs=[input, labels, input_length, label_length], outputs=[person_auth, ctc_loss]  ) 
    if weights and os.path.isfile(weights):
        model.load_weights(weights, by_name=True)
        logging.info( 'loaded world network weights') 
    compile_model(model, speaker) 
    test_func = K.function([input, labels, input_length, label_length, K.learning_phase()], [y_pred, ctc_loss, person_auth])
    return model,test_func

def liveness_auth_res3d(input_dim, output_dim, lipnet_weights):
    lipnet, test_func = lipnet_res3d(input_dim, output_dim, lipnet_weights) 
    person_feature = lipnet.get_layer(name='y_person_feature')
    auth = Dense(output_dim, kernel_initializer= "he_normal", 
            activation= "softmax", kernel_regularizer=l2(1e-4), name='y_person' )(person_feature)  
    liveness_net = Model(inputs=lipnet.inputs, outputs=[auth, lipnet.outputs[1]]) 
    compile_model(liveness_net) 
    return liveness_net, test_func
