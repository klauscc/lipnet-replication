from keras.models import Sequential,Model
from keras.layers import Input, Dense, Activation, Dropout, Convolution3D, MaxPooling3D, Flatten,ZeroPadding3D, TimeDistributed, SpatialDropout3D,BatchNormalization,Lambda
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
from .custom_layers import BiGRU 



class ModelLipNet(object):

    def __init__(self, verbose=True, compile_on_build=True, name='lipnet'):
        
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self._scaler = StandardScaler()
        self._complie_on_build = compile_on_build
        self.verbose = verbose
        self.name = name

    def fit(self, train_set, val_set, batch_size=32, nb_epoch=100):
        self.x_train = train_set['x']
        self.x_val = val_set['x']

        train_n = self.x_train.shape[0]
        val_n = self.x_val.shape[0]

        input_length=28
        input_length_train = np.zeros([train_n])
        input_length_train[:] = input_length
        input_length_val = np.zeros([val_n])
        input_length_val[:] = input_length

        label_length_train = np.ones([train_n])
        label_length_val = np.ones([val_n])

        #build model
        input_dim = self.x_train.shape[1:]
        output_dim = train_set['num_classes']
        self.build_model(input_dim, output_dim)

        self.y_train = np_utils.to_categorical(train_set['y'], output_dim)
        self.y_val = np_utils.to_categorical(val_set['y'], output_dim)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        checkpoint_path = './data/model/checkpoint/{}'.format(self.name)
        os.system('mkdir -p ' + checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)
        callbacks = [early_stopping, checkpoint]
        self.model.fit([self.x_train, self.y_train, input_length_train, label_length_train], [self.y_train, self.y_train], batch_size=batch_size, nb_epoch=nb_epoch, verbose=self.verbose, validation_data=([self.x_val, self.y_val, input_length_val, label_length_val], [self.y_val, self.y_val]))

    def build_model(self, input_dim, output_dim):

        input = Input(shape=input_dim)
        labels = Input(name='the_labels', shape=[output_dim], dtype='float32')
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
        bigru1 = BiGRU(stcnn3_maxpool_flatten, 512)
        #Bi-GRU-2
        bigru2 = BiGRU(bigru1, 512)

        #fc linear layer
        li = TimeDistributed(Dense(28))(stcnn3_maxpool_flatten)

        #flatten and to 0-9
        li_flatten = Flatten()(li)
        fc_clf = Dense(output_dim)(li_flatten)

        y_pred = Activation('softmax', name='y_pred')(fc_clf)
        y_pred_1 = Activation('softmax', name='y_pred_1')(li)

        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred_1, labels, input_length, label_length])

         
        self.model = Model(input=[input, labels, input_length, label_length], output=[loss_out,y_pred])

        if self._complie_on_build:
            optimizer = Adam(lr=0.0001)
            self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred,'y_pred':'categorical_crossentropy'},
                                optimizer=optimizer,
                                loss_weights={'ctc':1., 'y_pred':0.5},
                                metrics={'y_pred':'accuracy'})

    def ctc_lambda_func(self,args):
         y_pred, labels, input_length, label_length = args

         # the 2 is critical here since the first couple outputs of the RNN
         # tend to be garbage:
         y_pred = y_pred[:, 2:, :]
         return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

