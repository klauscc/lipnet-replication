import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.layers import Input, Dense, Activation, Dropout, Conv3D, MaxPooling3D, Flatten,ZeroPadding3D, TimeDistributed, SpatialDropout3D,BatchNormalization,Lambda,GRU,SpatialDropout1D, SpatialDropout2D, GlobalAveragePooling2D,AveragePooling2D,Conv2D
from model.resnet3d import basic_block, _bn_relu

def build_auth_net(lipnet):
    net = lipnet
    person_feature = net.get_layer('y_person_feature').output
    auth = Dense(2, kernel_initializer= "he_normal", 
        activation= "softmax", kernel_regularizer=l2(1e-4), name='y_person_2' )(person_feature)
    liveness_net = Model(inputs=net.inputs, outputs=[auth, net.get_layer('ctc').output ]) 

    print net.get_layer( 'y_pred') 
    test_func = K.function(net.inputs + [K.learning_phase(),], [net.get_layer('y_pred').output , auth])
    liveness_net.summary() 
    optimizer = Adam(lr=0.0001)
    liveness_net.compile(loss={'ctc': lambda y_true, y_pred: y_pred, 'y_person_2': 'categorical_crossentropy'},
                            optimizer=optimizer,
                            metrics= [ 'accuracy'] 
                            )
    return liveness_net, test_func

def build_auth_net_v3(input_dim, output_dim):
    input = Input(name='inputs', shape=input_dim)
    feature = shared_layers(input, input_dim) 
    #STCNN-4
    stcnn4_convolution = Conv3D(128, (3, 3, 3), strides=(1,3,3), kernel_initializer='he_uniform', name= 'stcnn4_convolution')(feature)
    stcnn4_bn = BatchNormalization(name= 'stcnn4_bn')(stcnn4_convolution)
    stcnn4_acti = Activation('relu', name= 'stcnn4_acti')(stcnn4_bn)
    #SPATIAL-DROPOUT
    stcnn4_dp = SpatialDropout3D(0.5, name= 'stcnn4_dp')(stcnn4_acti)
    #MAXPOOLING-3
    time_flatten = TimeDistributed(Flatten(), name= 'time_flatten')(stcnn4_dp) 
    auth_fc = TimeDistributed(Dense(output_dim, activation= 'relu'), name= 'auth_fc_'+str(output_dim))(time_flatten) 
    auth_flatten = Flatten(name= 'auth_flatten')(auth_fc) 
    # auth_fc = Dropout(0.5)(auth_fc)  
    auth_out = Dense(output_dim, kernel_initializer= 'he_uniform', activation='softmax', name= 'y_person_'+str(output_dim))(auth_flatten)
    model_auth = Model( inputs=input, outputs=auth_out) 
    optimizer = Adam(lr=0.0001)
    model_auth.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model_auth

def build_auth_net_v4(input_dim, output_dim):
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
    auth_out = Dense(output_dim, kernel_initializer= 'he_uniform', activation='softmax', name= 'y_person_'+str(output_dim))(auth_flatten)
    model_auth = Model( inputs=input, outputs=auth_out) 
    optimizer = Adam(lr=0.0001)
    model_auth.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model_auth

def build_auth_net_res_v2(input_dim, output_dim):
    """docstring for build_auth_net_res_v2"""
    kernel_regularizer = l2(1e-4) 

    input = Input(name='inputs', shape=input_dim)
    conv1 = Conv3D(32, (3,5,5), strides=(1,2,2), padding= 'same', kernel_initializer="he_normal", kernel_regularizer=kernel_regularizer)(input)  #90x25x50x32
    conv2 = basic_block(32)(conv1)

    conv3 = Conv3D(64, (3,3,3), strides=(1,2,2), padding= 'same', kernel_initializer="he_normal", kernel_regularizer=kernel_regularizer)(conv2)     # 90x12x25x64
    conv4 = basic_block(64)(conv3)  

    conv5 = Conv3D(96, (3,3,3), strides=(1,2,2), padding= 'same', kernel_initializer="he_normal", kernel_regularizer=kernel_regularizer)(conv4)     # 90x6x12x96
    conv6 = basic_block(96)(conv5)  

    # conv7 = Conv3D(128, (3,3,3), strides=(1,2,2), padding= 'same', kernel_initializer="he_normal", kernel_regularizer=kernel_regularizer)(conv6)     # 90x3x6x128
    conv7 = TimeDistributed(Conv2D(128, (3,3), strides=(2,2), padding= 'same', kernel_initializer= 'he_normal', kernel_regularizer=kernel_regularizer  ) )(conv6)
    conv8 = basic_block(256)(conv7)  
    block_out = _bn_relu(conv8) 
    average_pooling = TimeDistributed(AveragePooling2D(pool_size= (2,3) ) )(block_out) #90x256
    average_pooling = TimeDistributed(Flatten())(average_pooling) 
    average_pooling = TimeDistributed(Dropout(0.5) )(average_pooling) 
    hidden_1 = TimeDistributed(Dense(512, kernel_initializer= 'he_normal', activation= 'relu', kernel_regularizer=kernel_regularizer) ) (average_pooling) 
    hidden_1 = TimeDistributed(Dropout(0.5) )(hidden_1) 
    auth_out = TimeDistributed(Dense(output_dim, kernel_initializer= 'he_normal', activation= 'softmax', kernel_regularizer=kernel_regularizer  ),name= 'y_person_'+str(output_dim) )(hidden_1) #90xoutput_dim
    model_auth = Model( inputs=input, outputs=auth_out) 
    optimizer = Adam(lr=0.0001)
    model_auth.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # for layer in model_auth.layers[0:-5]:
        # layer.trainable=False
    return model_auth 
    

def shared_layers(input, input_dim):
    #STCNN-1
    stcnn1_padding = ZeroPadding3D(padding=(1,2,2), input_shape = input_dim, name= 'stcnn1_padding')(input) 
    stcnn1_convolution = Conv3D(32, (3, 5, 5), strides=(1,2,2), kernel_initializer='he_uniform', name= 'stcnn1_convolution')(stcnn1_padding)
    stcnn1_bn = BatchNormalization( name= 'stcnn1_bn')(stcnn1_convolution)
    stcnn1_acti = Activation('relu', name= 'stcnn1_acti')(stcnn1_bn)
    #SPATIAL-DROPOUT
    stcnn1_dp = SpatialDropout3D(0.5, name= 'stcnn1_dp')(stcnn1_acti)
    #MAXPOOLING-1
    stcnn1_maxpool = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name= 'stcnn1_maxpool')(stcnn1_dp)

    #STCNN-2
    stcnn2_padding = ZeroPadding3D(padding=(1,2,2), input_shape = input_dim, name= 'stcnn2_padding')(stcnn1_maxpool)
    stcnn2_convolution = Conv3D(64, (3, 5, 5), strides=(1,1,1), kernel_initializer='he_uniform', name= 'stcnn2_convolution')(stcnn2_padding)
    # stcnn2_convolution = Conv3D(64, (3, 5, 5), strides=(1,2,2), kernel_initializer='he_uniform')(stcnn2_padding)
    stcnn2_bn = BatchNormalization( name= 'stcnn2_bn')(stcnn2_convolution)
    stcnn2_acti = Activation('relu', name= 'stcnn2_acti')(stcnn2_bn)
    #SPATIAL-DROPOUT
    stcnn2_dp = SpatialDropout3D(0.5, name= 'stcnn2_dp')(stcnn2_acti)
    #MAXPOOLING-2
    stcnn2_maxpool = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name= 'stcnn2_maxpool')(stcnn2_dp)

    #STCNN-3
    stcnn3_padding = ZeroPadding3D(padding=(1,1,1), input_shape = input_dim, name= 'stcnn3_padding')(stcnn2_maxpool)
    stcnn3_convolution = Conv3D(96, (3, 3, 3), strides=(1,1,1), kernel_initializer='he_uniform', name= 'stcnn3_convolution')(stcnn3_padding)
    stcnn3_bn = BatchNormalization( name= 'stcnn3_bn')(stcnn3_convolution)
    stcnn3_acti = Activation('relu', name= 'stcnn3_acti')(stcnn3_bn)
    #SPATIAL-DROPOUT
    stcnn3_dp = SpatialDropout3D(0.5, name= 'stcnn3_dp')(stcnn3_acti)
    #MAXPOOLING-3
    stcnn3_maxpool = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name= 'shared_layers')(stcnn3_dp)
    return stcnn3_maxpool

