#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:56:52 2018

@author: ldk
"""

# 利用sklearn自建评价函数 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score 
from keras.callbacks import Callback 
class RocAucEvaluation(Callback): 
    def __init__(self, validation_data=(), interval=1): 
        super(Callback, self).__init__() 
        self.interval = interval 
        self.x_val,self.y_val = validation_data 
    
    def on_epoch_end(self, epoch, log={}): 
        if epoch % self.interval == 0: 
            y_pred = self.model.predict(self.x_val, verbose=0) 
            score = roc_auc_score(self.y_val, y_pred) 
            print('\n ROC_AUC - epoch:%d - score:%.6f \n' % (epoch+1, score)) 
            x_train,y_train,x_label,y_label = train_test_split(train_feature, train_label, train_size=0.95, random_state=233) 

RocAuc = RocAucEvaluation(validation_data=(y_train,y_label), interval=1) 
hist = model.fit(x_train, x_label, batch_size=batch_size, epochs=epochs, validation_data=(y_train, y_label), callbacks=[RocAuc], verbose=2)
