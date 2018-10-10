#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:55:20 2018

@author: ldk
"""

def shuffle_data(inputfile):
    import pandas as pd
    import numpy as np
    csv_input = pd.read_csv(inputfile,error_bad_lines = False)
##    print "csv_input.shape[0]",csv_input.shape[0]
    randArray = np.random.rand(csv_input.shape[0])
##    print "randArray.shape",randArray.shape
    csv_input['rand'] = randArray
    csv_input.sort_values(by = 'rand',axis = 0,inplace = True )#ÓÃÖµÅÅÐò½ÏÎª°²È«
    del csv_input['rand']
    csv_input.to_csv(inputfile, index=False,index_label = False ,mode = 'wb+')

###############################################################################


    
###############################################################################    
def read_testdata(data_file,params):
    import pandas as pd
    
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2,f_classif,mutual_info_classif,f_regression,mutual_info_regression,SelectPercentile,SelectFpr,SelectFdr,SelectFwe,GenericUnivariateSelect
    from sklearn.preprocessing import normalize, minmax_scale

    import NIRVERSION2_1
    
    data = pd.read_csv(data_file)
    #ÌØÕ÷Ñ¡È¡ºóµÄÐÂÊý¾Ý
    X = data.drop('label',axis = 1)
    X = X.drop('SampleDate',axis = 1)
    X = X.drop('picktime',axis = 1)
    X = X.drop('classes',axis = 1)
    X = X.drop('temp_label',axis = 1)    
    y = data.label
    print "data.label :y",y.shape

    ##É¸Ñ¡²¨³¤·¶Î§ÄÚ
    for wave in X.columns:
##        print wave
        if float(wave) <= params['wlengths'] or float(wave) >= params['wlengthe']:
            X = X.drop(wave,axis = 1)

    
    ##È¥³ýÌØÊâÀà±ð
    X,y = X[y != params['droplabel']],y[y != params['droplabel']]
##    print "X,y",X_new.shape,y.shape
    lambdstr = "lambda " + params['lambdastr']
    print "lambdstr",lambdstr
    X = X.apply(eval(lambdstr))
        
    ##lambda ±í´ïÊ½ºÜÅ£¡Á¶Ô°É È¡Ö¸Êý±ä»»£¬ÏêÇéÇë¿´²©
    ##Ñ¡Ôñ¸úµ±Ç°Ä£ÐÍÏàÍ¬µÄÌØÕ÷Ñ¡Ôñ²ÎÊý½øÐÐÌØÕ÷ÌáÈ¡
    print "test_params :" , params['selectfeature_params']
    selectmodel = SelectKBest(score_func = params['selectfeature_params']['score_func'], k = params['selectfeature_params']['k'])
    X_new = selectmodel.fit_transform(X, y)

    ##»òÊÇ¸ù¾Ý²âÊÔÊý¾Ý×ÔÉí½øÐÐÌØÕ÷ÌáÈ¡£¬²»½¨Òé      
##    X_new = SelectKBest(chi2,k = dimention).fit_transform(X, y)
##    print "X_new ***********",X_new.shape

    ##¹æ·¶»¯ÌØÕ÷ÊôÐÔÖµ
    if params['transform'] == 'none':
        print "transform = none"
        return X_new,y
    elif params['transform'] == 'norm':
        print "transform = norm"
        X_new = normalize(X_new, norm='l2')  
    elif params['transform'] == 'minmax':
        print "transform = minmax"
        X_new = minmax_scale(X_new)
        
    return X_new, y


###############################################################################    
def read_data_selectfeature(filesAll,preproAll,boolAll):
    import pandas as pd
    import numpy as np
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2,f_classif,mutual_info_classif,f_regression,mutual_info_regression,SelectPercentile,SelectFpr,SelectFdr,SelectFwe,GenericUnivariateSelect
    from sklearn.preprocessing import normalize, minmax_scale

    data = pd.read_csv(filesAll.getTrainDataFile())
    
    X = data.drop('label',axis = 1)
    X = X.drop('temp_label',axis = 1)
    X = X.drop('SampleDate',axis = 1)
    X = X.drop('picktime',axis = 1)
    X = X.drop('classes',axis = 1)
    columns = X.columns
    print columns
    print "columns length" , len(columns)
    difference = False
    if difference:
        X = X.T
    ##    print X
        X = X.diff(-1)   
        X = X.dropna().T  
    ##    print "dropna",X

        X = X.apply(lambda x:x-x.min())  #
    ##    X = minmax_scale(X)
        print "Scale shape",X.shape
        temp = columns[:-1]
    ##    print "temp",temp
        X = pd.DataFrame(X,columns = columns[:-1])  ##
    ##    X = np.r_[columns[:-1] , X]  
        print "columns",X
        
    y = data.label

    X,y = X[y != preproAll.getDropLabel()],y[y != preproAll.getDropLabel()]

##    print "%%%%%%%%X is :",X
##    print "wlengths is %d , wlengthe is :%d" %(wlengths,wlengthe)
    for wave in X.columns:
##        print wave
        wlengths,wlengthe = preproAll.getWave()
        if float(wave) <= wlengths or float(wave) >= wlengthe:
            X = X.drop(wave,axis = 1)            
##    print "*****####X_new is",X
            
    lambdastr = "lambda " + preproAll.getLamStr()
    print "lambda , Apply" , lambdastr,boolAll.getApplyOr()
    
    if boolAll.getApplyOr():
        X = X.apply(eval(lambdastr)) 
##        Xnew=Xnew.apply(lambda x: 10**x)   ##lambda ±í´ïÊ½ºÜÅ£¡Á¶Ô°É È¡Ö¸Êý±ä»»£¬ÏêÇéÇë¿´²©¿Í
##        Xnew = Xnew.apppy(lambda x:x)
        print "Applied Data",X

    print "F-mode :",preproAll.getFeaMode()
    chi2
    model = SelectKBest(chi2, k = preproAll.getFeaDimen())
    X_new_s = model.fit_transform(X, y)
    scores= model.scores_

    if preproAll.getFeaMode() == 'ANOVA':  ##ANOVA  f_classif: ANOVA F-value between label/feature for classification tasks
        model = SelectKBest(f_classif, k = preproAll.getFeaDimen)
        X_new_s = model.fit_transform(Xnew, y)
        scores= model.scores_

    ##³Ö¾Ã»¯ºÃµÄÌØÕ÷ÐÅÏ¢
    preproAll.setFeaParams(model.get_params(deep = True))
    print "trained Feaparams:",preproAll.getFeaParams()    

    ##³Ö¾Ã»¯ÌØÕ÷µÃ·Ö
    ScoreOfFea = pd.DataFrame(X.columns)
    ScoreOfFea['scores'] = scores.T
 
    Data_new = np.c_[X_new_s , y]  
    print "########****Data_new",Data_new
    
    train = Data_new[:int(len(Data_new)*0.7)]  
    test = Data_new[int(len(Data_new)*0.7):]

    train_y = train[:,-1:]  
    train_x = train[:,:-1]

    test_y = test[:,-1:]  
    test_x = test[:,:-1]

    if preproAll.getTransF() == 'none':
        print "transform = none"
        return train_x, train_y, test_x, test_y
    elif preproAll.getTransF() == 'norm':
        print "transform = norm"
        train_x = normalize(train_x, norm='l2')  
        test_x = normalize(test_x, norm='l2')
    elif preproAll.getTransF() == 'minmax':
        print "transform = minmax"
        train_x = minmax_scale(train_x)
        test_x = minmax_scale(test_x) 
        
    return train_x, train_y, test_x, test_y  