# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 23:29:30 2018

@author: ldk
"""
from sklearn.preprocessing import StandardScaler,MinMaxScaler,normalize
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest,chi2,f_classif
from sklearn.feature_selection import SelectFromModel
import pandas as pd
from sklearn.utils import shuffle 

def sampleBytarget(data,target,rate,replaceOr):
    print ("ori data shape",data.shape)
    targetdata = data[data['target']  == target]
    otherdata = data[data['target']  != target] 
    print ("targetdata shape",targetdata.shape)    
    print ("otherdata shape",otherdata.shape)
    
    k = int(rate*otherdata.shape[0])
    print ("k:",k)
    
    sampleddata = targetdata.sample(n=k, frac=None, replace=replaceOr, weights=None, random_state=7, axis=0)
    print ("sampleddata shape",sampleddata.shape)
    print ("sampleddata target",sampleddata[sampleddata['target'] == target].shape)
    sampleddata = sampleddata.append(otherdata)
    print ("after added sampled data shape",sampleddata.shape)    

    return sampleddata

def fature_select(train_x, train_y, K):

    
    model = SelectKBest(f_classif, k = K)
    X_train_s = model.fit_transform(train_x, train_y)    
    print ("X_train_s success",X_train_s.shape)
    
#    lsvc = LinearSVC(C=0.01, penalty="l2", dual=False).fit(train_x, train_y)
#    model = SelectFromModel(lsvc, prefit=True)
#    X_train_s = model.transform(train_x)
    
    return X_train_s


def drop_func(oriDF):
    predictors = [x for x in oriDF.columns if x not in ['CUST_NO', 'type',
                                                             'cc_open_time',	'cc_first_use_date','cc_last_use_date',
                                                             'dc_open_time',	'dc_first_use_date','dc_last_use_date','prod_no','ID'
                                                             ]]
    return oriDF[predictors]
    
def readdata(data_file,shuffleor,scalarOr,rate):

    data = pd.read_csv(data_file)#.fillna(0)#.replace('str',{-999:0}).replace('str',{'-999':0})     
    
    data = drop_func(data)
        
    if shuffleor:
        data = shuffle(data)
    
#    data = sampleBytarget(data,0,rate,False)

    print ("data shape",data.shape)
    print ("data columns",data.columns)
#    X = data.drop('phone_num',axis = 1)

#    X = fature_select(X, y,700)
    Train,Test= train_test_split(data, test_size=0.3, random_state=4)    
    Train.to_csv("./evaluation/trainData.csv",index = False, index_label = None,mode = 'w+')
    Test.to_csv("./evaluation/testData.csv",index = False, index_label = None,mode = 'w+')  


    train_x = Train.drop('target',axis = 1)
    train_y = Train.target 


    test_x = Test.drop('target',axis = 1)
    test_y = Test.target 
    

#    print train_x[:3]
#    print train_y[:3]
    
    
    return train_x, train_y, test_x, test_y 