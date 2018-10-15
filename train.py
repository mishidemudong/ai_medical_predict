#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:27:50 2018

@author: ldk
"""

from __future__ import print_function

from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import weightedRNNModel
import auc_ks_eval
import baseFileRead

def train(model,train_x, train_y):



    #iteration times
    epoch_1 = 120
    epoch_2 = 60
    eopch_3 = 18
    
    k = 1
    batch_size_3 = 1342*k
    batch_size_2 = 256*k
    batch_size_1 = 128*k

#    print ("step 1...."  )
#        model.summary()
    model.fit(train_x, train_y, epochs=epoch_1, batch_size=batch_size_1,shuffle = True,verbose=2) #4训练模型
        
#    print ("step 2....")
#    model.compile(loss='binary_crossentropy', optimizer=opt2 ,metrics=['binary_accuracy']) #binary_accuracy
#    model.fit(train_x, train_y, epochs=epoch_2, batch_size=batch_size_2,shuffle = True,verbose=2) #4训练模型
#
#    print ("step 3....")
#
#    model.compile(loss='binary_crossentropy', optimizer=opt3 ,metrics=['binary_accuracy'])
#    model.fit(train_x, train_y, epochs=eopch_3, batch_size=batch_size_3,shuffle = True,verbose=2) #4训练模型
    
    return model
 
    
def tf_train(model,train_x, train_y):
    

         
if __name__ == '__main__':
    
    
    #prepare train data
    train_x, train_y, test_x, test_y = baseFileRead.readdata(label_file,shuffle,scalarOr,rate)  
    print (train_x.shape[0])
    
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()
    
    onehottrain_y = tf.one_hot(np.asarray(train_y), 2).eval()
    
    wrnnmodel = weightedRNNModel.WRNNModel()
        
    print ("onehottrain_y",onehottrain_y[400:405])
    
    predict = wrnnmodel.predict_classes(test_x)
    print (test_y[:10].T)
    print (predict[:10])
    
    wrnnmodel.save(modelfile)
    
    auc_ks_eval.model_evaluation(wrnnmodel,train_x, train_y, test_x, test_y)
    
    sess.close()    
        

