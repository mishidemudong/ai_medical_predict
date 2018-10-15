# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 21:57:50 2018

@author: ldk
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,GaussianDropout,Embedding,Flatten,LSTM,SimpleRNN
from tensorflow.keras.layers import Dropout,BatchNormalization
from tensorflow.keras.constraints import MaxNorm as maxnorm  
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import Input
from tensorflow.keras.models import Model


import Attention_keras


class WRNNModel:
    def Average(self,inputs_list,weight_array):
        """Layer that averages a list of inputs.
    
        It takes as input a list of tensors,
        all of the same shape, and returns
        a single tensor (also of the same shape).
        """
        output = inputs_list[0] * weight_array[0]
        for i in range(1, len(inputs_list)):
            output += inputs_list[i] * weight_array[i]
        return output
            
    def Rnn_weightNet(self,inputs_list):
        '''
        Keras在layers包的recurrent模块中实现了RNN相关层模型的支持，并在wrapper模型上实现双向的RNN包装器.
        recurrent模块中的RNN模型包含RNN，LSTM,GRU等模型。       
        RNN：全连接RNNmoxing : SimpleRNN(units,activation='tanh',dropout=0.0,recurrent_dropout=0.0,return_sequences=False)     
        wrapper模块实现双向RNN模型：Bidirectional(layer,merge_mode='concat',weights=None)
        '''
        arr_len = len(inputs_list)
        RNNLayerList=[]
        for i in range(arr_len):
            tmpRnn = SimpleRNN(units=100,)(inputs_list[i]) 
            RNNLayerList.append(tmpRnn)
        
        MedicalFea_emb = K.concatenate(RNNLayerList, axis=1)
        
        return MedicalFea_emb
        
        
    
    def buildnet(self,trainX_dict, onehottrain_y):
        #prepare the embedding layer
        #for same type medical events         
        S1_input = Input(shape=(299, 299, 3), name='S1_input')
#        img2 = Input(shape=(299, 299, 3), name='img_2')
        
        inputs_list=[]
        S1_emb = Embedding(128, 100,input_length=trainX_dict['same1'].shape[1],embeddings_regularizer=regularizers.l2(0.00045))
        S2_emb = Embedding(128, 100,input_length=trainX_dict['same2'].shape[1],embeddings_regularizer=regularizers.l2(0.00045))
        #use  weights [0.35,0.65] to average
        
        inputs_list = inputs_list.append(S1_emb)
        inputs_list = inputs_list.append(S2_emb)
        
        WRNN_in = self.Rnn_weightNet(inputs_list)
        WRNN_out = Dense(len(inputs_list), activation='relu', name='Rnn_Weights')(WRNN_in)
        WRNN_model = Dense(2, activation='sigmoid', name='WRnn_Loss')(WRNN_out)
        self.weight_array = WRNN_model[-2].get_config('kernel_initializer') #get weights args of the last Dense layer
#        print ("weight_array",self.weight_array)
        #
        S_emb = self.Average(inputs_list,self.weight_array)
        
        #other type medical events
        O1_emb = Embedding(128, 100,input_length=trainX_dict['other1'].shape[1],embeddings_regularizer=regularizers.l2(0.00045))
        
        #concatenate different types
        MedicalFea_emb = K.concatenate([S_emb , O1_emb] , axis=1)
        
        #delta time feature , and the hyper weights Deltatime factor
        Delta_time_emb = Embedding(128, 20,input_length=trainX_dict['delta'].shape[1],embeddings_regularizer=regularizers.l1(0.00045))
        Dt_factor = 0.41
        
        #concatenate delta fea
        RNN_emb = K.concatenate([MedicalFea_emb , Dt_factor*Delta_time_emb] , axis=1)
        
        #build model        
        #define main_model input
        
        Main_input = Input(shape=(299, 299, 3), name='Main_input')
        
        Main_model = Sequential()
        Main_model.add(RNN_emb) #
        print ("1layer Embedding shape",Main_model.output_shape)
        
        Main_model.add(Flatten())
        
        ##add attention layer implement with keras
        Main_model.add(Attention_keras())
        
        #Lstm import hyperparameters 
        lstm_hid_size = 279
        Main_model.add(LSTM(units=lstm_hid_size, input_shape=(Main_model.output_shape[0], Main_model.output_shape[1]),dropout=0.2,return_sequences=True))

        Main_model.add(Dense(output_dim=lstm_hid_size, input_dim=lstm_hid_size,init='random_uniform',activation='tanh',use_bias=True,
                        kernel_regularizer=regularizers.l2(0.014),
#                        bias_regularizer=regularizers.l2(0.0003),
#                        activity_regularizer=regularizers.l2(0.003),
                        W_constraint=maxnorm(1)))
        Main_model.add(Dropout(0.4))
        Main_model.add(BatchNormalization())

        Main_model.add(Dense(output_dim = 2, input_dim=440, use_bias=True,
                        activation='sigmoid', 
                        W_constraint=maxnorm(1),name='Model_loss'))
        
        #binary task
        opt1 = Adagrad(lr=0.05, decay=1e-6)
        opt2 = Adagrad(lr=0.005, decay=1e-6)
        opt3 = Adagrad(lr=0.001, decay=1e-6)
        
        #multilable tasks
    #    opt3 = Adam(lr=0.001, decay=1e-6)
#        model.compile(loss='binary_crossentropy', optimizer=opt1 ,metrics=['binary_accuracy'])  ##auc,,loss:categorical_crossentropy  sparse_categorical_crossentropy,
        # compile the model (should be done *after* setting layers to non-trainable)
        Total_model = Model(inputs=[Main_input, S1_input], outputs=[Main_model, WRNN_model])
        Total_model.compile(optimizer=opt1,
                      loss={'WRnn_Loss': 'binary_crossentropy',
                            'Model_loss': 'binary_crossentropy'},
                      loss_weights={
                          'WRnn_Loss': 1.,       #####准备测试 两个命名空间相互赋值 比如 'ctg_out_1':'ctg_out_2' 前提维度必须相同
                          'Model_loss': 1.
                      },
                      metrics=['binary_accuracy'])
        
        return Total_model,Main_model



    

    
    
    
