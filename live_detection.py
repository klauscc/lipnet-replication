from model.lipnet import lipnet
from model.authentication import Resnet50

class LiveAuthentication(object):
    def __init__(lipnet_input_dim=(50,100, 3), authenet_input_dim=(200,200,3), lipnet_model='./data/checkpoints/lipnet_weights_multiuser.hdf5', authentication_model='./data/checkpoints/person_authentication_resnet50.hdf5'):
        self.authentication_net = Resnet50().build_net(input_dim = authenet_input_dim)
        self.lipnet = lipnet()

    def authentication():
