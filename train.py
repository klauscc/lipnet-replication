import numpy as np
import os
import sys
from dataset import DatasetLip
from model.lipnet import ModelLipNet

vsr = DatasetLip(re_generate=False)
[x_train,y_train,x_val,y_val] = vsr.load_data()
train_set = {'x':x_train,'y':y_train,'num_classes':10}
val_set = {'x':x_val,'y':y_val,'num_classes':10}


lipnet = ModelLipNet()
lipnet.fit(train_set,val_set,batch_size=50, nb_epoch=1000)

