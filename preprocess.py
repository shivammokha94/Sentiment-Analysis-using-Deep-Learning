#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 18:44:11 2019

@author: smokha
"""

import os
import pandas as pd
import numpy as np


curr_path = os.getcwd()
half_path_data = '/pretrain_data/dictionary.txt'     #Dictionary with sentences
half_path_label = '/pretrain_data/sentiment_labels.txt'  #Dictionary for labels
glove_path = '/pretrain_data/glove.6B/glove.6B.100d.txt'   #Glove 100 diamension vector representation   
trainin_data_dir = '/Users/smokha/Projects/Produvia/NLP/Sentiment_Analysis/pretrain_data/training_data/'  #Test, Validation and Train data directory

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


    
    def gloveloader(glove_path):
 
#Load stanford GloVe vector data       
        print("Loading GloVe model from %s" % glove_path)
        file = open(os.path.join(curr_path+glove_path), 'r', encoding = 'utf-8')       
        model = {}
        
        for line in file:
            splitvec = line.split()
            word = splitvec[0]
            embedding = np.array([float(val) for val in splitvec[1:]])
            model[word] = embedding       
        print("Completed. Total", len(model), "words loaded")     
            
        return model
        
    
    def data_split(data, splitperc, trainin_data_dir):
        
#Split data into test, train and validation sets        
        masker = np.random.rand(len(data)) < splitperc
        train = data[masker]
        test_val = data[~masker]
        
        masker_val = np.random.rand(len(test_val)) < 0.6
        test = data[masker_val]
        val = data[~masker_val]
        
        return train, test, val
        