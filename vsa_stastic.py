#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import editdistance 

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__) ) 
RESULT_DIR = os.path.join(CURRENT_PATH, './data/vsa_result') 
# RESULT_DIR = os.path.join(CURRENT_PATH, './data/vsa_result_model_selected_by_val_loss') 

def stastic_target_speak_matching_sentence(target_df, ed_threshold, speaker_auth):
    """
    stastic the tatio of the target speaker speaks the matching sentence
    Args:
    ------
    target_df: pandas.DataFrame.values, numpy array. the df save the predicted result. all colums contain:`index,edit_distance,predict_str,source_str,speaker,speaker_accepted,speaker_auth`
    ed_threshold: float. The threshold 

    Returns:
    --------
    total: int. the total samples in df
    correct_count: int. the predicted count of the target speaker speaking the correct sentence
    ratio: float. correct_count / total
    """
    total = len(target_df) 
    correct_count = 0
    for idx, row in enumerate(target_df):
        if row[6] > speaker_auth and row[1] < ed_threshold:
            correct_count += 1
    ratio = float(correct_count)/ total 
    return total, correct_count, ratio

def stastic_target_speak_mismatching_sentence(target_df, ed_threshold, speaker_auth):
    """
    stastic the target speaker speaks the mismatching sentence

    Args:
    ------
    target_df: pandas.DataFrame. the df save the predicted result. all colums contain:`index,edit_distance,predict_str,source_str,speaker,speaker_accepted,speaker_auth`
    ed_threshold: float. The threshold 

    Returns:
    --------
    total: int. the total samples in df
    correct_count: int. the predicted count of the target speaker speaking the correct sentence
    ratio: float. correct_count / total
    """
    correct_count = 0
    total = 0
    for idx, row in enumerate(target_df) :
        sentence = row[3] 
        for idx_2, row_2 in enumerate(target_df) :
            if idx_2 == idx:
                continue
            else:
                total += 1
                med = editdistance.eval(row_2[2], sentence ) / float(len(sentence))
                if not (row_2[6] > speaker_auth and med < ed_threshold):
                    correct_count += 1
    ratio = float(correct_count)/ total 
    return total, correct_count, ratio

def stastic_imposter_speak_matching_sentence(imposter_df, ed_threshold, speaker_auth):
    """stastic the accuracy of imposter speaks the matching sentence
    Args:
    ------
    target_df: pandas.DataFrame. the df save the predicted result. all colums contain:`index,edit_distance,predict_str,source_str,speaker,speaker_accepted,speaker_auth`
    ed_threshold: float. The threshold 

    Returns:
    --------
    total: int. the total samples in df
    correct_count: int. the predicted count of the target speaker speaking the correct sentence
    ratio: float. correct_count / total
    
    """
    total = len(imposter_df) 
    correct_count = 0
    for idx, row in enumerate(imposter_df) :
        if row[6] < speaker_auth or row[1] > ed_threshold:
            correct_count += 1
    ratio = float(correct_count)/ total 
    return total, correct_count, ratio

def stastic_imposter_speak_mismatching_sentence(target_df, imposter_df, ed_threshold, speaker_auth):
    """docstring for stastic_imposter_speak_mismatching_sentence
    Args:
    ------
    target_df: pandas.DataFrame. the df save the predicted result. all colums contain:`index,edit_distance,predict_str,source_str,speaker,speaker_accepted,speaker_auth`
    ed_threshold: float. The threshold 

    Returns:
    --------
    total: int. the total samples in df
    correct_count: int. the predicted count of the target speaker speaking the correct sentence
    ratio: float. correct_count / total
    """
    correct_count = 0
    total = 0
    for idx, row in enumerate(target_df) :
        sentence = row[3] 
        for idx_2, row_2 in enumerate(imposter_df) :
            if row_2[3] == sentence:
                continue
            else:
                total += 1
                med = editdistance.eval(row_2[2], sentence ) / float(len(sentence))
                if not (row_2[6] > speaker_auth and med < ed_threshold):
                    correct_count += 1
    ratio = float(correct_count)/ total 
    return total, correct_count, ratio

    

def stastic_speaker(speaker,idx="", ed_threshold=0.3, speaker_auth=0.97):
    target_csv = os.path.join(RESULT_DIR, 'S{}_target_{}.csv'.format(speaker, idx) ) 
    imposter_csv = os.path.join(RESULT_DIR, 'S{}_imposter_{}.csv'.format(speaker, idx) ) 
    target_df = pd.read_csv(target_csv).values
    imposter_df = pd.read_csv(imposter_csv).values
    result = ()
    result = stastic_target_speak_matching_sentence(target_df, ed_threshold=ed_threshold, speaker_auth=speaker_auth) 
    result += stastic_target_speak_mismatching_sentence(target_df, ed_threshold, speaker_auth) 
    result += stastic_imposter_speak_matching_sentence(imposter_df, ed_threshold, speaker_auth) 
    result += stastic_imposter_speak_mismatching_sentence(target_df, imposter_df, ed_threshold, speaker_auth) 
    print "speaker {}, idx {}: {}".format(speaker, idx, result)
    return result

if __name__ == '__main__':
    # end = 35
    begin = 22
    end = 35
    # for speaker in range(begin,end):
        # stastic_speaker(speaker) 
    # speaker = 22
    # idx = '25_1'
    import sys
    speaker = int(sys.argv[1])
    # stastic_speaker(speaker, idx='0') 
    for n in [25, 50, 100, 200]:
    # for n in [200]:
        ave_result = np.zeros(4) 
        for idx in [ '{}_{}'.format(n,i) for i in range(1,6) ]: 
            result = stastic_speaker(speaker, idx, speaker_auth=0.97) 
            for i in range(4):
                ave_result[i] += result[2+3*i]  
        ave_result /= 5
        print( 'speaker {} training sample {}: {}'.format(speaker, n, ave_result) ) 
