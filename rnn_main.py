#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:40:42 2019

@author: smokha
"""

import sys
import lstm
import os
from tensorflow import keras
from nltk import RegexpTokenizer
import utility_functions
import numpy as np
import pandas as pd
curr_path = os.getcwd()
model_path = '/model/mdl_wts.hdf5'
glove_path = '/pretrain_data/glove.6B/glove.6B.100d.txt'   #Glove 100 diamension vector representation   
data_path = '/pretrain_data/training_data/main_data.csv'



class rnn_run():
  
    
    
    
    def train_model():
        
#Run LSTM RNN model    
        glove_dim = 100
        batch_size = 2000
        no_of_epochs = 25
    
        x_train, y_train, x_test, y_test, x_val, y_val, wt_matrix, max_len = lstm.model_runner.get_model_data()  #Get data
        model = lstm.model_runner.rnn_model(glove_dim, wt_matrix, max_len)  #Define model
    
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        mcp_save = keras.callbacks.ModelCheckpoint(os.path.join(curr_path + model_path), save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

        model.fit(x_train, y_train, batch_size=batch_size, epochs=no_of_epochs, validation_data=(x_val, y_val), callbacks = [earlyStopping, mcp_save, reduce_lr_loss]) 
        
        #Test model on test set
        model_score, model_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
        print("Accuracy on test set:", model_accuracy)   #Accuracy
        print("Score on test set:", model_score)  #Mean loss
        
        return model
     
 
    
    def get_data():
        
#Get max length and word matrix   
        data = pd.DataFrame(pd.read_csv((os.path.join(curr_path + data_path))))
        max_len = utility_functions.preprocess_data.len_sequencer(data)
        wt_matrix, word_index = utility_functions.preprocess_data.load_embeddings(os.path.join(curr_path + glove_path))   #load word embeddings and matrix
        
        return max_len, word_index


    
    def senti_analyze(model, data, max_len, word_index):
        
#Analyze sample data sentiment        
        print("Processing sample data to feed to tensorflow")
        
        pred_ls = []
        pred_ip_matrix = np.zeros((max_len, 1))
        
        tokenizer = RegexpTokenizer(r'\w+')
        
        data = data.lower()
        data = tokenizer.tokenize(data)
        
        ls = []
        for word in data:
            if word in word_index:
                ls.append(word_index[word])
            else:
                ls.append(0)
        
        temp_matrix = np.zeros(max_len)
        temp_matrix[: len(ls)] = ls
        temp_matrix = temp_matrix.astype(int)
        pred_ls.append(temp_matrix)
        pred_ip_matrix = np.asarray(pred_ls)
        
        score_matrix = model.predict(pred_ip_matrix, batch_size=1, verbose=0)
        score = np.round(np.argmax(score_matrix)/10, decimals = 2)
              
        #Aggregate of top 3 scores
        index_3 = np.argsort(score_matrix)[0][-3:]
        scores_3 = score_matrix[0][index_3]
        weights = scores_3/np.sum(scores_3)
        aggregate_score_3 = np.round(np.dot(index_3, weights)/10, decimals = 2)
        
        return score_matrix, score, aggregate_score_3
    
    
    def main():
        
        model = None
        try:
            model = keras.models.load_model(os.path.join(curr_path + model_path))
        except:
            print("Model not found, staring model run.....")
        
        if model is None:
            model = rnn_run.train_model()
            
        max_len, word_index = rnn_run.get_data()
        
#       sentence = input('Type in sentence (Length greater than 3 words):')
        sentence = '! Wow!'
        sc_matrix, get_prediction, aggre_pred = rnn_run.senti_analyze(model, sentence, max_len, word_index)

        if (aggre_pred >=0 and aggre_pred < 0.2):
            sent = 'Very negative'
        elif (aggre_pred >= 0.2 and aggre_pred < 0.4):
            sent = 'Negative'
        elif (aggre_pred >= 0.4 and aggre_pred < 0.6):
            sent = 'Neutral'
        elif (aggre_pred >= 0.6 and aggre_pred < 0.8):
            sent = 'Positive'
        elif (aggre_pred >= 0.8 and aggre_pred <=1):
            sent = 'Very positive'
        
        print("Sentence:", sentence)
        print("Sentiment:", sent, "with prediction probability:", aggre_pred, get_prediction)


    if __name__ == '__main__':
        main()
        
  
#
#lr = []
#indexing_matrix = np.zeros((max_len, 1))
#len(ls)
#padded_array = np.zeros(56) 
#padded_array[:len(ls)] = ls.astype(int)
#data_index_np_pad = padded_array.astype(int)
#lr.append(data_index_np_pad)
#indexing_matrix = np.asarray(lr)
#type(indexing_matrix)
#type(padded_array)
