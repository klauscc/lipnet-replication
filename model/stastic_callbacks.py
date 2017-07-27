import os
import numpy as np
import cairocffi as cairo
import editdistance 
from keras.callbacks import Callback

from common.utils.spell import Spell
from core.ctc import decode_batch 

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','common','dictionaries','grid.txt')

def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

class StatisticCallback(Callback):

    """used to statistic the accuracy after each epoch"""

    def __init__(self, test_func, log_savepath,test_data_gen, test_num, checkpoint_dir=None):
        self.test_func = test_func
        self.test_data_gen=test_data_gen
        self.log_savepath = log_savepath
        self.test_num = test_num
        self.checkpoint_dir=checkpoint_dir
        self.best_loss = 10000
        self.spell = Spell(path=PREDICT_DICTIONARY)
        Callback.__init__(self)
        with open(log_savepath, 'w+') as f:
            f.write('epoch,loss,val_loss,sentence_error_rate,word_error_rate,character_error_rate,edit_distance\n')

    def statistic(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        word_count = 0
        word_err_count = 0
        result_true = 0
        losses = []
        while num_left > 0:
            word_batch = next(self.test_data_gen)[0]
            num_proc = min(word_batch['inputs'].shape[0], num_left)
            input_list = [word_batch['inputs'][0:num_proc], word_batch['labels'][0:num_proc], word_batch['input_length'][0:num_proc], word_batch['label_length'][0:num_proc], 0]
            decoded_res, batch_loss = decode_batch(self.test_func, input_list)
            batch_loss = np.mean(batch_loss) 
            losses.append(batch_loss)
            for j in range(0, num_proc):
                decoded_res[j] = self.spell.sentence(decoded_res[j]) 
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])

                #sentense level accuracy
                #print(decoded_res[j], "| ground truth: ",word_batch['source_str'][j] )
                if  decoded_res[j] == word_batch['source_str'][j]:
                    result_true += 1
                #wer
                reference = word_batch['source_str'][j].split()
                predicted = decoded_res[j].split()
                word_count += len(reference)
                word_err_count += wer(reference, predicted)

                #edit distance
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['labels'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num #the same as cer
        mean_ed = mean_ed / num
        ser = float(num-result_true) / num
        word_error_rate = float(word_err_count) / word_count
        mean_loss = np.mean(losses)
        print('\nOut of %d samples: Sentence Error Rate: %.3f WER: %0.3f Mean normalized edit distance(CER): %0.3f Mean edit distance: %.3f loss: %.3f\n'
              % (num, ser, word_error_rate, mean_norm_ed, mean_ed, mean_loss))
        return (num, ser, word_error_rate, mean_norm_ed, mean_ed, mean_loss)

    def on_epoch_end(self, epoch, logs={}):
        #statistic every 3 epoch for the first 30 epochs
        if epoch < 30 and epoch % 3 != 0:
            return
        num, ser, wer, cer, mean_ed, mean_loss = self.statistic(self.test_num)
        if mean_loss < self.best_loss:
            if self.checkpoint_dir:
                print ("\nnew val_loss {} less than previous best val_loss{}, saving weight to {}\n".format(mean_loss, self.best_loss, self.checkpoint_dir))
                checkpoint_dir_base, ext = os.path.splitext(self.checkpoint_dir)
                save_dir = "{}-{}-{}-{}{}".format(checkpoint_dir_base, epoch, mean_loss, cer, ext)
                self.model.save_weights(save_dir)
            self.best_loss = mean_loss
        with open(self.log_savepath, "a") as f:
            f.write("{},{:.5},{:.5},{:.3},{:.3},{:.3},{:.3}\n".format(epoch, logs.get('loss'), mean_loss, ser, wer, cer, mean_ed))
