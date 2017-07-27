import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def limitMemory():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    # config.gpu_options.per_process_gpu_memory_fraction=0.55
    set_session(tf.Session(config=config))


def init():
    limitMemory()
