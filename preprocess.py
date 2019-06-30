#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 18:44:11 2019

@author: smokha
"""

import os
import pandas as pd

curr_path = os.getcwd()
half_path_data = '/pretrain_data/dictionary.txt'     #Dictionary with sentences
half_path_label = '/pretrain_data/sentiment_labels.txt'  #Dictionary for labels
    
    

class preprocess():


    def gettraindata():
    
#Process sentence data
        sentence_data = pd.DataFrame(pd.read_table(os.path.join(curr_path + half_path_data)))    #Read data
        sentence_data = sentence_data['sentence|index'].str.split('|', expand=True)    #Split data from |
        sentence_data = sentence_data.rename(columns = {0: 'sentence', 1: 'index'})    #Rename columns

#Process label data
        sentence_labels = pd.DataFrame(pd.read_table(os.path.join(curr_path + half_path_label)))    #Read data
        sentence_labels = sentence_labels['phrase ids|sentiment values'].str.split('|', expand=True) #Split data from |
        sentence_labels = sentence_labels.rename(columns = {0: 'index', 1: 'prob_label'})    #Rename columns

        sentence_merge = sentence_data.merge(sentence_labels, how = 'inner', on='index')   #Merge data
        
        return sentence_merge