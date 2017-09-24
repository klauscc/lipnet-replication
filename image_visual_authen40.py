from __future__ import print_function
from vis.losses import ActivationMaximization,Loss
from vis.optimizer import *
from vis import backend, backprop_modifiers
from vis.utils import utils
import os
from vis.visualization import overlay,saliency
from keras import activations
import PIL.Image as pil
from matplotlib import pyplot as plt
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing import image
import matplotlib.cm as cm
from keras.models import Sequential, Model
from keras.layers import *

from gridSinglePersonAuthentication import GRIDSinglePersonAuthentication
from authenbase_40_data_generator import Authenbase40DataGenerator

from model.lipnet_res3d import lipnet_res3d
from model.auth_net import build_auth_net_res_v2
import cv2
def my_model():
    input_dim = (90, 50, 100, 3) 
    output_dim = 2
    max_label_length = 50
    speaker = 2
    # weights_path = './data/checkpoints_grid/grid_vsa_speaker_22.hdf5'
    weights_path = './data/checkpoints/authenbase40_vsa_res_v2_speaker_2.hdf5'

    pos_n = 25
    # data_generator = GRIDSinglePersonAuthentication(auth_person=speaker, pos_n=pos_n)
    data_generator = Authenbase40DataGenerator(target_speaker=speaker)
    # model,test_func = lipnet_res3d(input_dim=input_dim, output_dim=max_label_length, weights = None, speaker= speaker)
    model = build_auth_net_res_v2(input_dim, output_dim, viz=True) 
    model.load_weights(weights_path) 
    model.summary() 

    # lip_path = './data/GRID/lips/s22/prahzs'
    # mini_batch_lips = data_generator.gen_batch(begin=0, 
            # batch_size=1,
            # paths=(lip_path,) ,
            # gen_words=False,
            # auth_person=speaker,
            # scale=1/255.) 
    batch_size = 2
    gen = data_generator.next_batch(batch_size, phase= 'test', shuffle=True) 
    inputs,outputs = next(gen) 
    # print (outputs)
    # Swap softmax with linear, only needed when visualing softmax layer
    layer_idx = utils.find_layer_idx(model, 'y_person_2')
    # model.layers[layer_idx].activation = activations.linear
    # model = utils.apply_modifications(model)
    inputs = inputs[0] 

    visualize_saliency_3Dcnn(model, layer_idx, filter_indices=1, seed_input=inputs,
                             original_img=inputs*255,backprop_modifier= None)
# def test():
    # # Build the VGG16 network with ImageNet weights
    # model = VGG16(weights='imagenet', include_top=True)

    # # Utility to search for layer index by name.
    # # Alternatively we can specify this as -1 since it corresponds to the last layer.
    # layer_idx = utils.find_layer_idx(model, 'predictions')

    # # Swap softmax with linear
    # model.layers[layer_idx].activation = activations.linear
    # model = utils.apply_modifications(model)

    # plt.rcParams['figure.figsize'] = (18, 6)

    # img1 = utils.load_img('images/ouzel1.jpg', target_size=(224, 224))
    # img2 = utils.load_img('images/ouzel2.jpg', target_size=(224, 224))

    # # f, ax = plt.subplots(1, 2)
    # # ax[0].imshow(img1)
    # # ax[1].imshow(img2)

    # f, ax = plt.subplots(1, 2)

    # for i, img in enumerate([img1, img2]):
        # # 20 is the imagenet index corresponding to `ouzel`
        # # heatmap = saliency.visualize_cam(model, layer_idx, filter_indices=20, seed_input=img,backprop_modifier='guided')
        # heatmap = saliency.visualize_saliency(model, layer_idx, filter_indices=20, seed_input=img,backprop_modifier=None)
        # print (np.shape(heatmap))
        # # Lets overlay the heatmap onto original image.
        # ax[i].imshow(overlay(heatmap,img))

    # plt.show()


def _identity(x):
    return x

def deprocess_input(input_array, input_range=(0, 255)):
    """Utility function to scale the `input_array` to `input_range` throwing away high frequency artifacts.

    Args:
        input_array: An N-dim numpy array.
        input_range: Specifies the input range as a `(min, max)` tuple to rescale the `input_array`.

    Returns:
        The rescaled `input_array`.
    """
    # normalize tensor: center on 0., ensure std is 0.1
    input_array = input_array.copy()
    input_array -= input_array.mean()
    input_array /= (input_array.std() + K.epsilon())
    input_array *= 0.1

    # clip to [0, 1]
    input_array += 0.5
    input_array = np.clip(input_array, 0, 1)

    # Convert to `input_range`
    return (input_range[1] - input_range[0]) * input_array + input_range[0]

class Optimizer_3Dcnn(Optimizer):
    def __init__(self):
        super(Optimizer_3Dcnn,self).__init__()

    def _get_seed_input(self, seed_input):
        """Creates a random `seed_input` if None. Otherwise:
            - Ensures batch_size dim on provided `seed_input`.
            - Shuffle axis according to expected `image_data_format`.
        """
        desired_shape = (1, ) + K.int_shape(self.input_tensor)[1:]
        print (desired_shape)
        if seed_input is None:
            return utils.random_array(desired_shape, mean=np.mean(self.input_range),
                                      std=0.05 * (self.input_range[1] - self.input_range[0]))

        # Add batch dim if needed.
        if len(seed_input.shape) != len(desired_shape):
            seed_input = np.expand_dims(seed_input, 0)

        # Only possible if channel idx is out of place.
        if seed_input.shape != desired_shape:
            seed_input = np.moveaxis(seed_input, -1, 1)
        # for i in range(np.shape(seed_input)[0]):
        #     x = seed_input[i,...]
        #     seed_input[i,...] = x
        return seed_input.astype(K.floatx())

    def minimize(self, seed_input=None, max_iter=200,
                 input_modifiers=None, grad_modifier=None,
                 callbacks=None, verbose=True):
        """Performs gradient descent on the input image with respect to defined losses.

        Args:
            seed_input: An N-dim numpy array of shape: `(samples, channels, image_dims...)` if `image_data_format=
                channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
                Seeded with random noise if set to None. (Default value = None)
            max_iter: The maximum number of gradient descent iterations. (Default value = 200)
            input_modifiers: A list of [InputModifier](vis.input_modifiers#inputmodifier) instances specifying
                how to make `pre` and `post` changes to the optimized input during the optimization process.
                `pre` is applied in list order while `post` is applied in reverse order. For example,
                `input_modifiers = [f, g]` means that `pre_input = g(f(inp))` and `post_input = f(g(inp))`
            grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
                specify anything, gradients are unchanged. (Default value = None)
            callbacks: A list of [OptimizerCallback](vis.callbacks#optimizercallback) instances to trigger.
            verbose: Logs individual losses at the end of every gradient descent iteration.
                Very useful to estimate loss weight factor(s). (Default value = True)

        Returns:
            The tuple of `(optimized input, grads with respect to wrt, wrt_value)` after gradient descent iterations.
        """
        seed_input = self._get_seed_input(seed_input)
        input_modifiers = input_modifiers or []
        grad_modifier = _identity if grad_modifier is None else get(grad_modifier)

        callbacks = callbacks or []
        cache = None
        best_loss = float('inf')
        best_input = None

        grads = None
        wrt_value = None

        for i in range(max_iter):
            # Apply modifiers `pre` step
            for modifier in input_modifiers:
                seed_input = modifier.pre(seed_input)

            # 0 learning phase for 'test'
            computed_values = self.compute_fn([seed_input, 0])
            losses = computed_values[:len(self.loss_names)]
            named_losses = zip(self.loss_names, losses)
            overall_loss, grads, wrt_value = computed_values[len(self.loss_names):]

            # TODO: theano grads shape is inconsistent for some reason. Patch for now and investigate later.
            if grads.shape != wrt_value.shape:
                grads = np.reshape(grads, wrt_value.shape)

            # Apply grad modifier.
            grads = grad_modifier(grads)

            # Trigger callbacks
            # for c in callbacks:
            #     c.callback(i, named_losses, overall_loss, grads, wrt_value)

            # Gradient descent update.
            # It only makes sense to do this if wrt_tensor is input_tensor. Otherwise shapes wont match for the update.
            if self.wrt_tensor is self.input_tensor:
                step, cache = self._rmsprop(grads, cache)
                seed_input += step

            # Apply modifiers `post` step
            for modifier in reversed(input_modifiers):
                seed_input = modifier.post(seed_input)

            if overall_loss < best_loss:
                best_loss = overall_loss.copy()
                best_input = seed_input.copy()
            print ('best_input',np.shape(best_input))
            for i in range(np.shape(best_input)[1]):
                best_input[0,i,...] = deprocess_input(best_input[0,i,...],self.input_range)

        # Trigger on_end
        # for c in callbacks:
        #     c.on_end()
        return best_input[0], grads, wrt_value
        # return deprocess_input(best_input[0], self.input_range), grads, wrt_value

def visualize_saliency_with_losses(input_tensor, losses, seed_input,original_img, grad_modifier='absolute'):
    opt = Optimizer(input_tensor, losses, norm_grads=False)
    grads = opt.minimize(seed_input=seed_input, max_iter=1, grad_modifier=grad_modifier, verbose=False)[1]
    # print (np.shape(grads))

    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    grads = np.max(grads, axis=channel_idx)
    # if not os.path.exists('./image'):
    #     os.mkdir('./images')

    print (np.shape(grads))
    for i in range(np.shape(grads)[1]):
        temp_grads = utils.normalize(grads[:,i,...])
        temp_grads[temp_grads < 0.2] = 0
        # print ('temp_grads',np.shape(temp_grads))
        heatmap = np.uint8(cm.jet(temp_grads)[..., :3] * 255)[0]
        # heatmap = cv2.medianBlur(heatmap,5) 
        # print (heatmap)
        # heatmap[heatmap < 255] = 0 
        img = original_img[i,...]

        temp = image.array_to_img(overlay(heatmap, img,alpha=0.5))
        pil.Image.save(temp,'images/overlay{}.jpg'.format(i))

        temp = image.array_to_img(heatmap)
        pil.Image.save(temp,'images/heatmap{}.jpg'.format(i))

        temp = image.array_to_img(img)
        pil.Image.save(temp,'images/original{}.jpg'.format(i))
    # Normalize and create heatmap.

class ActivationMaximization3D(Loss):
    """A loss function that maximizes the activation of a set of filters within a particular layer.

    Typically this loss is used to ask the reverse question - What kind of input image would increase the networks
    confidence, for say, dog class. This helps determine what the network might be internalizing as being the 'dog'
    image space.

    One might also use this to generate an input image that maximizes both 'dog' and 'human' outputs on the final
    `keras.layers.Dense` layer.
    """
    def __init__(self, layer, filter_indices):
        """
        Args:
            layer: The keras layer whose filters need to be maximized. This can either be a convolutional layer
                or a dense layer.
            filter_indices: filter indices within the layer to be maximized.
                For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.

                If you are optimizing final `keras.layers.Dense` layer to maximize class output, you tend to get
                better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
                output can be maximized by minimizing scores for other classes.
        """
        super(ActivationMaximization, self).__init__()
        self.name = "ActivationMax3D Loss"
        self.layer = layer
        self.filter_indices = utils.listify(filter_indices)

    def build_loss(self):
        layer_output = self.layer.output

        # For all other layers it is 4
        is_dense = K.ndim(layer_output) == 2
        # is_dense = K.ndim(layer_output) == 3

        loss = 0.
        for idx in self.filter_indices:
            if is_dense:
                loss += -K.mean(layer_output[:, :, idx])
            else:
                # slicer is used to deal with `channels_first` or `channels_last` image data formats
                # without the ugly conditional statements.
                loss += -K.mean(layer_output[utils.slicer[:, idx, ...]])

        return loss

def visualize_saliency_3Dcnn(model, layer_idx, filter_indices, seed_input,original_img,
                       backprop_modifier=None, grad_modifier='absolute'):

    if backprop_modifier is not None:
        modifier_fn = backprop_modifiers.get(backprop_modifier)
        # model = backend.modify_model_backprop(model, 'guided')
        model = modifier_fn(model)


    # `ActivationMaximization` loss reduces as outputs get large, hence negative gradients indicate the direction
    # for increasing activations. Multiply with -1 so that positive gradients indicate increase instead.
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), -1)
    ]
    # losses = [
        # (ActivationMaximization3D(model.layers[layer_idx], filter_indices), -1)
    # ]
    visualize_saliency_with_losses(model.input, losses, seed_input,original_img, grad_modifier)


if __name__ == '__main__':

    # test()
    # print (globals())
    my_model()
