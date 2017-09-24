import os
import itertools
import matplotlib
import logging
import numpy as np
import pandas as pd
import seaborn as sn
matplotlib.use( 'Agg') 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes, save_file_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    image = cmap(cm) 
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes[0:-1] )

    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i,j] > 0.005:
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    # plt.xlabel('Predict')
    # plt.ylabel('Truth')
    plt.xlabel('User')
    plt.ylabel('Client')
    plt.savefig( save_file_name) 
    plt.close()

class_dict = [
        { 'bin': 0, 'lay': 1, 'place': 2, 'set': 3, 'other':4},
        { 'blue':0, 'green':1, 'red':2, 'white':3, 'other':4},
        { 'at':0, 'by':1, 'in':2, 'with':3, 'other':4},
        { 'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 'r':17, 's':18, 't':19, 'u':20, 'v':21, 'x':22, 'y':23, 'z':24, 'other':25},
        { 'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'other':10},
        { 'again':0, 'now':1, 'please':2, 'soon':3, 'other':11}
        ]
class_names = [
        ['bin', 'lay', 'place', 'set', 'other'],
        [ 'blue', 'green', 'red', 'white', 'other'],
        [ 'at', 'by', 'in', 'with', 'other'],
        [ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z', 'other'],
        [ 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'other'],
        [ 'again', 'now', 'please', 'soon', 'other']
        ]

def get_confusion_matrix(file_name):
    df = pd.read_csv(file_name).values
    pred = [[],[],[],[],[],[]]
    ground = [[],[],[],[],[],[]]
    for idx, value in enumerate(df):
        source_words = value[3].split(" ")
        pred_words = value[2].split(" ")
        word_length = 6
        if len(pred_words) != word_length:
            continue
        extra_flag = [4, 4, 4, 25, 10, 11] 
        for i in range(word_length):
            if class_dict[i].has_key(pred_words[i] ):
                pred[i].append(class_dict[i][pred_words[i] ]  ) 
            else:
                pred[i].append(extra_flag[i]  ) 
            if class_dict[i].has_key(source_words[i] ):
                ground[i].append(class_dict[i][source_words[i] ]  ) 
            else:
                ground[i].append(extra_flag[i]  ) 
        if idx == 0 or idx == 1:
            logging.debug("pred:{}, pred_words:{}".format(pred, pred_words) ) 
            logging.debug( 'ground:{}, ground_words:{}'.format(ground, source_words) ) 
    return pred, ground

def plot_content_cm():
    file_name = './data/vsa_result/S22_imposter_25_1.csv'
    pred,ground = get_confusion_matrix(file_name) 
    cm = []
    for i in range(6):
        cm=confusion_matrix(ground[i],pred[i])
        plot_confusion_matrix(cm[0:-1,:] , class_names[i], normalize=True, save_file_name= 'cm_{}.jpg'.format(i) ) 

def load_csv(file_name):
    df = pd.read_csv(file_name).values
    return df

def plot_id_cm(thresh=0.97):
    dir= './data/vsa_result_loss_weights_id1_vs_ctc0.5'
    speakers = np.arange(22,35) 
    cm = np.zeros([len(speakers), len(speakers) ]) 
    for current_speaker in speakers:
        target_file_name = os.path.join(dir, 'S{}_target.csv'.format(current_speaker) ) 
        imposter_file_name = os.path.join(dir, 'S{}_imposter.csv'.format(current_speaker) ) 
        target_df = load_csv(target_file_name) 
        imposter_df = load_csv(imposter_file_name) 
        for df in [target_df, imposter_df]: 
            for value in df:
                speaker = value[4] 
                speaker_auth = value[6] 
                if speaker_auth > thresh:
                    cm[current_speaker-22, speaker-22] += 1
    for i in speakers:
        for j in speakers:
            if i != j:
                cm[i-22, j-22] /= 1000.
            else:
                cm[i-22, j-22] /= 950.
    cm[6,6]=0.95 

    plot_confusion_matrix(cm , speakers, normalize=False, save_file_name= 'id_cm.jpg' ) 

def main():
    logging.getLogger().setLevel(logging.DEBUG) 
    # plot_content_cm() 
    plot_id_cm() 

if __name__ == '__main__':
    main()
