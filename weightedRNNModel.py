# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 21:57:50 2018

@author: ldk
"""


#import tensorflow as tf
#import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.advanced_activations import ThresholdedReLU
from keras.layers import Dense ,GaussianDropout,Embedding,Flatten,LSTM
from keras.layers import Dropout,BatchNormalization
from keras.constraints import maxnorm
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.objectives import categorical_crossentropy

from params import Hyperparas


class WRNNModel:
    def Average(self,trainX_dict,same_dict,weight_array):
        """Layer that averages a list of inputs.
    
        It takes as input a list of tensors,
        all of the same shape, and returns
        a single tensor (also of the same shape).
        """
        inputs_list=[]
        S1_emb = Embedding(128, 100,input_length=trainX_dict[same_dict['same1']].shape[1],embeddings_regularizer=regularizers.l2(0.00045))
        S2_emb = Embedding(128, 100,input_length=trainX_dict[same_dict['same2']].shape[1],embeddings_regularizer=regularizers.l2(0.00045))
        #use  weights [0.35,0.65] to average
        
        inputs_list = inputs_list.append(S1_emb)
        inputs_list = inputs_list.append(S2_emb)
        
        output = inputs_list[0] * weight_array[0]
        for i in range(1, len(inputs_list)):
            output += inputs_list[i] * weight_array[i]
        return output
    
    def AttentionLayer(self,inputlayer):
        
    
    def buildnet(self,trainX_dict, onehottrain_y):
        #prepare the embedding layer
        #for same type medical events 

        
        self.weight_array = [0.35,0.65] 
        S_emb = self.Average(trainX_dict,self.weight_array)
        
        #other type medical events
        O1_emb = Embedding(128, 100,input_length=trainX_dict['other1'].shape[1],embeddings_regularizer=regularizers.l2(0.00045))
        
        #concatenate different types
        MedicalFea_emb = K.concatenate([S_emb , O1_emb] , axis=1)
        
        #delta time feature
        Delta_time_emb = Embedding(128, 20,input_length=trainX_dict['delta'].shape[1],embeddings_regularizer=regularizers.l1(0.00045))
        
        #concatenate delta fea
        RNN_emb = K.concatenate([MedicalFea_emb , Delta_time_emb] , axis=1)
        
        #build model
        model = Sequential()
        model.add(RNN_emb) #
        print ("1layer Embedding shape",model.output_shape)
        
        model.add(Flatten())
        
        model.add(LSTM(128, input_shape=(model.output_shape[0], model.output_shape[1]),return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(128,return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(output_dim=570, input_dim=620,init='random_uniform',activation='tanh',use_bias=True,
                        kernel_regularizer=regularizers.l2(0.014),
#                        bias_regularizer=regularizers.l2(0.0003),
#                        activity_regularizer=regularizers.l2(0.003),
                        W_constraint=maxnorm(1)))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        model.add(Dense(output_dim = 2, input_dim=440, use_bias=True,
                        activation='sigmoid', 
                        W_constraint=maxnorm(1)))
        
        #binary task
        opt1 = Adagrad(lr=0.05, decay=1e-6)
        opt2 = Adagrad(lr=0.005, decay=1e-6)
        opt3 = Adagrad(lr=0.001, decay=1e-6)
        
        #multilable tasks
        opt3 = Adam(lr=0.001, decay=1e-6)
        #iteration times
        epoch_1 = 120
        epoch_2 = 60
        eopch_3 = 18
        
        k = 1
        batch_size_3 = 1342*k
        batch_size_2 = 256*k
        batch_size_1 = 128*k

        print ("step 1...."  )
#        model.summary()
        model.compile(loss='binary_crossentropy', optimizer=opt1 ,metrics=['binary_accuracy'])  ##auc,,loss:categorical_crossentropy  sparse_categorical_crossentropy,

        model.fit(train_x, onehottrain_y, epochs=epoch_1, batch_size=batch_size_1,shuffle = True,verbose=2) #4训练模型
            
        print ("step 2....")
        model.compile(loss='binary_crossentropy', optimizer=opt2 ,metrics=['binary_accuracy']) #binary_accuracy
        model.fit(train_x, onehottrain_y, epochs=epoch_2, batch_size=batch_size_2,shuffle = True,verbose=2) #4训练模型

        print ("step 3....")

        model.compile(loss='binary_crossentropy', optimizer=opt3 ,metrics=['binary_accuracy'])
        model.fit(train_x, onehottrain_y, epochs=eopch_3, batch_size=batch_size_3,shuffle = True,verbose=2) #4训练模型
        
        return model

    

    
    
    
