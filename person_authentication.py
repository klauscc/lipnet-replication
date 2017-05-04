from gridPersonAuthentication import GRIDPersonAuthentication
from model.authentication import Resnet50
from configurations import init
from keras.callbacks import ModelCheckpoint
import os


init()

batch_size = 32
nb_epoch = 100
weight_savepath = './data/checkpoints/person_authentication_resnet50.hdf5'
initial_epoch = 0
input_dim = (200,200,3)

gridPerson = GRIDPersonAuthentication(target_size = (input_dim[0],input_dim[1]), debug=False)
train_generator = gridPerson.next_batch(batch_size, phase='train', shuffle = True)
train_steps = gridPerson.train_num // batch_size
val_generator = gridPerson.next_batch(batch_size, phase='val', shuffle= False, random_transform=False)
val_steps = gridPerson.val_num // batch_size

checkpointer = ModelCheckpoint(filepath=weight_savepath,save_best_only=True,save_weights_only=True)

net = Resnet50().build_net()
net.summary()
if os.path.isfile(weight_savepath):
    print ("load weight file:{}".format(weight_savepath))
    net.load_weights(weight_savepath)
net.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
net.fit_generator(generator = train_generator, steps_per_epoch = train_steps,
                epochs = nb_epoch, initial_epoch = initial_epoch,
                workers = 2,
                pickle_safe = True,
                validation_data=val_generator, validation_steps=val_steps,
                callbacks = [checkpointer]
            )
