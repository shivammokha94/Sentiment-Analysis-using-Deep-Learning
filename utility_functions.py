#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 18:44:11 2019

@author: smokha
"""

import os
import pandas as pd
import numpy as np
from nltk import RegexpTokenizer, tokenize

curr_path = os.getcwd()
half_path_data = '/pretrain_data/dictionary.txt'     #Dictionary with sentences
half_path_label = '/pretrain_data/sentiment_labels.txt'  #Dictionary for labels
glove_path = '/pretrain_data/glove.6B/glove.6B.100d.txt'   #Glove 100 diamension vector representation   

class preprocess_data():


    def getdata():
    
        print("Retriving data and merging")
#Process sentence data
        sentence_data = pd.DataFrame(pd.read_table(os.path.join(curr_path + half_path_data)))    #Read data
        sentence_data = sentence_data['sentence|index'].str.split('|', expand=True)    #Split data from |
        sentence_data = sentence_data.rename(columns = {0: 'sentence', 1: 'index'})    #Rename columns

#Process label data
        sentence_labels = pd.DataFrame(pd.read_table(os.path.join(curr_path + half_path_label)))    #Read data
        sentence_labels = sentence_labels['phrase ids|sentiment values'].str.split('|', expand=True) #Split data from |
        sentence_labels = sentence_labels.rename(columns = {0: 'index', 1: 'prob_label'})    #Rename columns

        sentence_merge = sentence_data.merge(sentence_labels, how = 'inner', on='index')   #Merge data   
        print("Complete")
        
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

        
    
    def data_split(data, split_percentage_test_train, split_percentage_test_val, trainin_data_dir):
        
#Split data into test, train sets        
        print("Splitting test, train and validation sets")
        masker = np.random.rand(len(data)) < split_percentage_test_train
        train = data[masker]
        test_val = data[~masker]

#Split test and validation sets        
        masker_val = np.random.rand(len(test_val)) < split_percentage_test_val
        test = test_val[masker_val]
        val = test_val[~masker_val]
   
#Reset indices     
        train.reset_index()
        test.reset_index()
        val.reset_index()

#Save data to CSV        
        train.to_csv(os.path.join(trainin_data_dir + 'train_data.csv'))
        test.to_csv(os.path.join(trainin_data_dir + 'test_data.csv'))
        val.to_csv(os.path.join(trainin_data_dir + 'validation_data.csv'))
        print("Split complete")
        
        return train, test, val

 

    def load_embeddings(emb_path):
   
#Read data from glove path and convert it into a weight matrix and a dictionary of indices & words        
        print("Loading word embeddings from from %s" % emb_path)
        file = open(os.path.join(curr_path+glove_path), 'r', encoding = 'utf-8')
        word_index = {}
        wt_vec = []
        
        for line in file:
            word, vec = line.split(u' ', 1)
            word_index[word.lower()] = len(wt_vec)
            wt_vec.append(np.array(vec.split(),  dtype=np.float32))
        wt_vec.append(np.random.uniform(-0.05, 0.05, wt_vec[0].shape))
        print("Embedding load complete") 
        return np.stack(wt_vec), word_index



    def len_sequencer(data):
        
#Get the longest sentence word count     
        print("Running length sequencer to get max length")
        max_len = 0
        
        for index, row in data.iterrows():
            sentence = row['sentence']           
            if len(sentence.split()) > max_len:
                max_len = len(sentence.split())
        print("Maximum length found:", max_len)
        
        return max_len
   

    
    def tensor_vec_pipline(data, word_index, max_len):
        
#Create data maxtrix to be fed to the keras model       
        print("Creating data to feed to tensorflow")
        df_len = len(data)
        indexing_matrix = np.zeros((df_len, max_len), dtype = 'int32')
        r_inc = 0

        tokenizer = RegexpTokenizer(r'\w+')

        for index, row in data.iterrows():
    
            sentence = row['sentence']
            sen_tokenize = tokenizer.tokenize(sentence)
    
            c_inc = 0
            
            for word in sen_tokenize:
                try:
                    indexing_matrix[r_inc][c_inc] = word_index[word]
                except Exception as e:
                    #print(e, word)
                    if (str(e) == word):
                        indexing_matrix[r_inc][c_inc] = 0
                continue
        
            c_inc = c_inc + 1
        r_inc = r_inc + 1
        print("Run complete")
        
        return indexing_matrix
    


    def getylabels(data):
        
#Convert data into categories labels and return a matrix of dummy variables
        print("Converting labels")
        values = data['prob_label']
        values = values.astype(float)
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1]  #Bin ranges
        categories = [0, 1, 2, 3, 4]  #Creation of 5 categories -> Very negative, Negative, Neutral, Positive, Very Positive
        
        values = pd.cut(values, bins = bins, labels = categories)
        cat_dum = pd.get_dummies(values, prefix='', prefix_sep='')
        label_matrix = cat_dum.as_matrix()
        print("Label conversion complete")
        
        return label_matrix
        
    
    
    
#  TEST
#main_data = pd.DataFrame(preprocess.getdata())     #Read and merge labels and dictionary
#main_data.to_csv(os.path.join(trainin_data_dir + 'main_data.csv'))  #Save data to folder
#wt_matrix, word_index = preprocess.load_embeddings(glove_path)   #load word embeddings and matrix
#train_data, test_data, val_data = preprocess.data_split(main_data, 0.8)   #Split train, test and validation data and save to CSV file
#max_len = preprocess.len_sequencer(main_data)   #Get max seuence length
#x_train = preprocess.tensor_vec_pipline(train_data)   #Get feature matrix for training data
#x_val = preprocess.tensor_vec_pipline(val_data)         #Get feature matrix for validation data
#x_test = preprocess.tensor_vec_pipline(test_data)       #Get feature matrix for test data
#y_train = preprocess.getylabels(train_data)    #Get y category matrix for training data
#y_val = preprocess.getylabels(val_data)         #Get y category matrix for validation data
#y_test = preprocess.getylabels(test_data)       #Get y category matrix for test data
#

