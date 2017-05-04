import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def limitMemory():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))


def init():
    limitMemory()
