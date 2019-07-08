#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 13:53:03 2019

@author: smokha
"""

import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import utility_functions
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

trainin_data_dir = '/Users/smokha/Projects/Produvia/NLP/Sentiment_Analysis/pretrain_data/training_data/'  #Test, Validation and Train data directory
glove_path = '/pretrain_data/glove.6B/glove.6B.100d.txt'   #Glove 100 diamension vector representation   


class model_runner():

    
    def get_model_data():

#Preprocess before getting the model running        
        print("Perparing data")
        main_data = pd.DataFrame(utility_functions.preprocess_data.getdata())     #Read and merge labels and dictionary
        main_data.to_csv(os.path.join(trainin_data_dir + 'main_data.csv'))  #Save data to folder
    
        wt_matrix, word_index = utility_functions.preprocess_data.load_embeddings(glove_path)   #load word embeddings and matrix

        train_test_split_perc = 0.8
        test_val_split_perc = 0.6
        train_data, test_data, val_data = utility_functions.preprocess_data.data_split(main_data, train_test_split_perc, test_val_split_perc, trainin_data_dir)   #Split train, test and validation data and save to CSV file

        max_len = utility_functions.preprocess_data.len_sequencer(main_data)   #Get max seuence length

        x_train = utility_functions.preprocess_data.tensor_vec_pipline(train_data, word_index, max_len)   #Get feature matrix for training data
        x_val = utility_functions.preprocess_data.tensor_vec_pipline(val_data, word_index, max_len)         #Get feature matrix for validation data
        x_test = utility_functions.preprocess_data.tensor_vec_pipline(test_data, word_index, max_len)       #Get feature matrix for test data

        y_train = utility_functions.preprocess_data.getylabels(train_data)    #Get y category matrix for training data
        y_val = utility_functions.preprocess_data.getylabels(val_data)         #Get y category matrix for validation data
        y_test = utility_functions.preprocess_data.getylabels(test_data)       #Get y category matrix for test data
        
        print("Data summary")
        print("Train size:", x_train.shape, ". Count of Labels:", len(y_train))
        print("Validation size:", x_val.shape, ". Count of Labels:", len(y_val))
        print("Test size:", x_test.shape, ". Count of Labels:", len(y_val))
        print("Classes:", np.unique(y_train.shape[1]))
        print("End of summary")
        return x_train, y_train, x_test, y_test, x_val, y_val, wt_matrix, max_len



    def rnn_model(emb_dia_len, wt_matrix, max_len):
        
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(len(wt_matrix), emb_dia_len, weights = [wt_matrix], input_length=max_len, trainable = False))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
        model.add(keras.layers.Dense(512, activation=0.5))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10, activation=10))
        #Compile model
        model.compile(loss='binary_crossentropy', optimizer='adan', metrics=['accuracy'])
        
        return model
    
    
    
    
    
    
#    
#    
#### Hyper paramater tuning    
#x, y, x_t, y_t, x_val, y_val, wtmx, ml = model_runner.get_model_data()    
#
## fix random seed for reproducibility
#seed = 7
#np.random.seed(seed)
## load pima indians dataset
#
## create model
#model = KerasClassifier(build_fn=model_runner.rnn_model, verbose=0)
## grid search epochs, batch size and optimizer
#optimizers = ['rmsprop', 'adam']
#init = ['glorot_uniform', 'normal', 'uniform']
#epochs = [50, 100, 150]
#batches = [5, 10, 20]
#param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
#grid = GridSearchCV(estimator=model, param_grid=param_grid)
#grid_result = grid.fit(x_t, y_t)
## summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))