from keras.callbacks import Callback
import os
import keras.backend as K
from keras.utils import np_utils
import itertools
from common.utils.spell import Spell
import numpy as np

def ctc_lambda_func(args):
     y_pred, labels, input_length, label_length = args

     # the 2 is critical here since the first couple outputs of the RNN
     # tend to be garbage:
     y_pred = y_pred[:, 2:, :]
     return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def decode_batch(test_func, input_list):
    out, loss,auth = test_func(input_list)

    # using tensorflow ctc decoder
    y_pred = K.placeholder(shape=[out.shape[0],out.shape[1]-2,out.shape[2]])
    input_length_value = np.zeros(out.shape[0])
    input_length_value[:] = out.shape[1]-2
    input_length = K.placeholder(shape=[out.shape[0]])
    decoder = K.ctc_decode(y_pred, input_length, beam_width=200, greedy=False)
    decoded = K.get_session().run(decoder, feed_dict={y_pred:out[:,2:], input_length: input_length_value})[0][0]

    ret = []
    for j in range(out.shape[0]):
        outstr= ''
        for c in decoded[j]:
            if c >= 0 and c < 26:
                outstr += chr(c + ord('a'))
            elif c == 26:
                outstr += ' '
        ret.append(outstr)

    # greedy search
    # for j in range(out.shape[0]):
        # out_best = list(np.argmax(out[j, 2:], 1))
        # out_best = [k for k, g in itertools.groupby(out_best)]
        # 26 is space, 27 is CTC blank char
        # outstr = ''
        # for c in out_best:
            # if c >= 0 and c < 26:
                # outstr += chr(c + ord('a'))
            # elif c == 26:
                # outstr += ' '
        # ret.append(outstr)
    return ret, loss, auth
