import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers import Dense, TimeDistributed,Dropout, Flatten

def build_auth_net(lipnet):
    net = lipnet
    person_feature = net.get_layer('y_person_feature').output
    auth = Dense(2, kernel_initializer= "he_normal", 
            activation= "softmax", kernel_regularizer=l2(1e-4), name='y_person_auth' )(person_feature)  
    liveness_net = Model(inputs=net.inputs, outputs=[auth, net.get_layer('ctc').output ]) 

    print net.get_layer( 'y_pred') 
    test_func = K.function(net.inputs + [K.learning_phase(),], [net.get_layer('y_pred').output , auth])
    liveness_net.summary() 
    optimizer = Adam(lr=0.0001)
    liveness_net.compile(loss={'ctc': lambda y_true, y_pred: y_pred, 'y_person_auth': 'categorical_crossentropy'},
                            optimizer=optimizer,
                            metrics= [ 'accuracy'] 
                            )
    return liveness_net, test_func

def build_auth_net_v2(lipnet):
    net = lipnet
    shared_feature_layers = net.get_layer( 'shared_features_flatten').output
    auth_hidden = TimeDistributed( Dense(2, kernel_initializer= 'he_normal') )(shared_feature_layers) 
    auth_hidden = TimeDistributed(Dropout(0.5) )(auth_hidden)  
    auth_flatten = Flatten()(auth_hidden) 
    auth = Dense(2, kernel_initializer= "he_normal", 
            activation= "softmax", kernel_regularizer=l2(1e-4), name='y_person_auth' )(auth_flatten)  
    liveness_net = Model(inputs=net.inputs, outputs=[auth, net.get_layer('ctc').output ]) 
    test_func = K.function(net.inputs + [K.learning_phase(),], [net.get_layer('y_pred').output , auth])
    liveness_net.summary() 
    optimizer = Adam(lr=0.0001)
    liveness_net.compile(loss={'ctc': lambda y_true, y_pred: y_pred, 'y_person_auth': 'categorical_crossentropy'},
                            optimizer=optimizer,
                            metrics= [ 'accuracy'] 
                            )
    return liveness_net, test_func
