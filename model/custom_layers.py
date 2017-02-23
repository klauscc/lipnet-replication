from keras.layers.core import Layer
import keras.backend as K
from keras.layers import GRU, merge


def BiGRU(x, output_dim, return_sequences=True, init='glorot_uniform', inner_init='orthogonal', activation='tanh', inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0, name='BiGRU'):
    x_reverse = K.reverse(x, 0)
    if output_dim % 2 != 0:
        raise ValueError('the `output_dim` argument should be an even')

    single_output = output_dim //2
    branch1 = GRU(single_output, init, inner_init, activation, inner_activation, W_regularizer, U_regularizer, b_regularizer, dropout_W, dropout_W, name=(name+'-1'), return_sequences=return_sequences)(x)
    branch2 = GRU(single_output, init, inner_init, activation, inner_activation, W_regularizer, U_regularizer, b_regularizer, dropout_W, dropout_W, name =(name +'-2'), return_sequences=return_sequences)(x_reverse)
    output = merge([branch1, branch2], mode='concat', concat_axis=1, name = (name + '-merge'))
    return output

# class SpatialDropout3D(Layer):
    # def __init__(self, **kwargs):
        # super(SpatialDropout3D, self).__init__(**kwargs)

    # def call(self, x, mask=None):
        # return K
