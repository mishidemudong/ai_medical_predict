# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:59:48 2017

@author: liang
"""
def dense_to_one_hot(labels_dense, num_classes):
    import numpy
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
    
def onehotencode(target, num_classes):
    import tensorflow as tf    
    import numpy as np
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    
    sess = tf.InteractiveSession(config=config)
    print("onehotencoder start....")
    with sess.as_default():
        tmponehot =  tf.one_hot(np.asarray(target.T)[0] , num_classes).eval()
    print("onehotencoder done....")
    sess.close()
    return tmponehot
  
class DataSet(object):
    from tensorflow.python.framework import dtypes
    def __init__(self,
               data,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32):
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = data.shape[0]
    
    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        import numpy
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]

def readdata(data_file,shuffle, one_hot):
    import pandas as pd
    import numpy as np
    import random
    from sklearn.preprocessing import normalize
    from tensorflow.contrib.learn.python.learn.datasets import base
    
    data = pd.read_csv(data_file)
    #特征选取后的新数据
    
    X = data.drop('label',axis = 1)
    X = X.drop('SampleDate',axis = 1)
    X = X.drop('picktime',axis = 1)
    X = X.drop('classes',axis = 1)
    X = X.drop('temp_label',axis = 1)    
    y = data.label
    print("data.label :y",y.shape)
    
    for wave in X.columns:
##        print wave
        if float(wave) <= 495 or float(wave) >= 980:
            X = X.drop(wave,axis = 1)
            
    ##normalize the data
    norm = True
    if norm:
        X = normalize(X, norm='l2')

    Data_new = np.c_[X , y]  
    print("########****Data_new",Data_new)
    
        
    if shuffle:
        random.shuffle(Data_new)
    
    validation_size = 0.1
    
    train = Data_new[:int(len(Data_new)*0.8)]  
    print("train size",len(train))
    test = Data_new[int(len(Data_new)*0.8):]
    
    validation = train[:int(len(train)*validation_size)]
    print("validation size",len(validation))
    train = train[int(len(train)*validation_size):]    
    print("subtrain size",len(train))
    train_y = train[:,-1:]  
    train_x = train[:,:-1]

    validation_y = validation[:,-1:]  
    validation_x = validation[:,:-1] 
    
    test_y = test[:,-1:]  
    test_x = test[:,:-1]
    
    num_classes = 4
    if one_hot:
       train_y = onehotencode(train_y, num_classes)
       validation_y = onehotencode(validation_y, num_classes)
       test_y = onehotencode(test_y, num_classes) 
       
    train_ = DataSet(train_x, train_y)
    validation_ = DataSet(validation_x,validation_y)
    test_ = DataSet(test_x, test_y)
    
    print("test.test_x",test_.data)

    return base.Datasets(train=train_, validation=validation_, test=test_)
