from model.authentication import *
from configurations import init
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import os

from ourLipPersonAuthentication import OurLipPersonAuthentication as db


init()

batch_size = 256
nb_epoch = 100
initial_epoch = 0
input_dim = (50, 100, 3)

gridPerson = db(data_dir='./data/ourlip' ,target_size = (input_dim[0],input_dim[1]), debug=False)
train_generator = gridPerson.next_batch(batch_size, phase='train', shuffle = True)
train_steps = gridPerson.train_num // batch_size
print ( "train_steps : {}".format(train_steps) ) 
# val_generator = gridPerson.next_batch(batch_size, phase='val', shuffle= True, random_transform=False)
val_generator = gridPerson.next_batch(batch_size, phase='val', shuffle = True)
val_steps = gridPerson.val_num // batch_size
print ( "val_steps : {}".format(val_steps) ) 


# weight_savepath = './data/checkpoints/person_authentication_resnet50.hdf5'
# checkpointer = ModelCheckpoint(filepath='./data/checkpoints/person_authentication_resnet50-{epoch:02d}-{val_loss:.2f}-{val_acc:.4f}.hdf5',save_best_only=True,save_weights_only=True, verbose=1)
# net = Resnet50().build_net()

weight_savepath = ''
checkpointer = ModelCheckpoint(filepath='./data/checkpoints/ourlip_person_authentication_my_model1-{epoch:02d}-{val_loss:.2f}-{val_acc:.4f}.hdf5',save_best_only=True,save_weights_only=True, verbose=1)
net = My_model1().build_net(input_dim = input_dim, output_dim=200)  
# net = Jianguo_model().build_net( input_dim=input_dim)

net.summary()
if os.path.isfile(weight_savepath):
    print ("load weight file:{}".format(weight_savepath))
    net.load_weights(weight_savepath)
# optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
optimizer = 'rmsprop'
net.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
net.fit_generator(generator = train_generator, steps_per_epoch = train_steps,
                epochs = nb_epoch, initial_epoch = initial_epoch,
                validation_data=val_generator, validation_steps=val_steps,
                callbacks = [checkpointer]
            )
