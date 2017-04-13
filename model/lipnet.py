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
from keras.callbacks import Callback
import itertools
import cairocffi as cairo
import editdistance 
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
    test_func = K.function([input, labels, input_length, label_length, K.learning_phase()], [y_pred, loss_out])
    model.summary()
    return model,test_func

def ctc_lambda_func(args):
     y_pred, labels, input_length, label_length = args

     # the 2 is critical here since the first couple outputs of the RNN
     # tend to be garbage:
     y_pred = y_pred[:, 2:, :]
     return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def decode_batch(test_func, input_list):
    out, loss = test_func(input_list)

    # y_pred = K.placeholder(shape=[out.shape[0],73,28])
    # input_length_value = np.zeros(out.shape[0])
    # input_length_value[:] = 73
    # input_length = K.placeholder(shape=[out.shape[0]])
    # decoder = K.ctc_decode(y_pred, input_length, greedy=True)
    # decoded = K.get_session().run(decoder, feed_dict={y_pred:out[:,2:], input_length: input_length_value})[0][0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        # 26 is space, 27 is CTC blank char
        outstr = ''
        # outstr1= ''
        # for c in decoded[j]:
            # if c >= 0 and c < 26:
                # outstr1 += chr(c + ord('a'))
            # elif c == 26:
                # outstr1 += ' '

        for c in out_best:
            if c >= 0 and c < 26:
                outstr += chr(c + ord('a'))
            elif c == 26:
                outstr += ' '
        #print outstr, " ",outstr1
        ret.append(outstr)
    return ret, np.mean(loss)

def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

class StatisticCallback(Callback):

    """used to statistic the accuracy after each epoch"""

    def __init__(self, test_func, log_savepath,test_data_gen, test_num, checkpoint_dir=None):
        self.test_func = test_func
        self.test_data_gen=test_data_gen
        self.log_savepath = log_savepath
        self.test_num = test_num
        self.checkpoint_dir=checkpoint_dir
        self.best_loss = 10000
        Callback.__init__(self)
        with open(log_savepath, 'w+') as f:
            f.write('epoch,loss,val_loss,sentence_error_rate,word_error_rate,character_error_rate,edit_distance\n')

    def statistic(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        word_count = 0
        word_err_count = 0
        result_true = 0
        losses = []
        while num_left > 0:
            word_batch = next(self.test_data_gen)[0]
            num_proc = min(word_batch['inputs'].shape[0], num_left)
            input_list = [word_batch['inputs'][0:num_proc], word_batch['labels'][0:num_proc], word_batch['input_length'][0:num_proc], word_batch['label_length'][0:num_proc], 0]
            decoded_res, batch_loss = decode_batch(self.test_func, input_list)
            losses.append(batch_loss)
            for j in range(0, num_proc):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])

                #sentense level accuracy
                #print(decoded_res[j], "| ground true: ",word_batch['source_str'][j] )
                if  decoded_res[j] == word_batch['source_str'][j]:
                    result_true += 1
                #wer
                reference = word_batch['source_str'][j].split()
                predicted = decoded_res[j].split()
                word_count += len(reference)
                word_err_count += wer(reference, predicted)

                #edit distance
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['labels'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num #the same as cer
        mean_ed = mean_ed / num
        ser = float(num-result_true) / num
        word_error_rate = float(word_err_count) / word_count
        mean_loss = np.mean(losses)
        print('\nOut of %d samples: Sentence Error Rate: %.3f WER: %0.3f Mean normalized edit distance(CER): %0.3f Mean edit distance: %.3f loss: %.3f'
              % (num, ser, word_error_rate, mean_norm_ed, mean_ed, mean_loss))
        return (num, ser, word_error_rate, mean_norm_ed, mean_ed, mean_loss)
    def on_epoch_end(self, epoch, logs={}):
        num, ser, wer, cer, mean_ed, mean_loss = self.statistic(self.test_num)
        if mean_loss < self.best_loss:
            if self.checkpoint_dir:
                print ("\nnew val_loss {} less than previous best val_loss{}, saving weight to {}".format(mean_loss, self.best_loss, self.checkpoint_dir))
                self.model.save_weights(self.checkpoint_dir)
            self.best_loss = mean_loss
        with open(self.log_savepath, "a") as f:
            f.write("{},{:.5},{:.5},{:.3},{:.3},{:.3},{:.3}\n".format(epoch, logs.get('loss'), mean_loss, ser, wer, cer, mean_ed))
