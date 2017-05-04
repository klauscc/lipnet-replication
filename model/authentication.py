from keras.applications.resnet50 import ResNet50
from keras.layers import Input, GlobalAveragePooling2D,Dense
from keras.models import Model
class Resnet50():
    def build_net(self,input_dim=(200,200,3),output_dim=34,weights = 'imagenet'):
        input_tensor = Input(shape=input_dim)
        base_model = ResNet50(input_tensor = input_tensor, weights=weights,include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        out = Dense(output_dim, activation='softmax', name='out')(x)
        model = Model(input=base_model.input, output = out)
        return model
