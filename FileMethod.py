# -*- coding: gbk -*-
import NIRVERSION2_1

def shuffle_data(inputfile):
    import pandas as pd
    import numpy as np
    csv_input = pd.read_csv(inputfile,error_bad_lines = False)
##    print "csv_input.shape[0]",csv_input.shape[0]
    randArray = np.random.rand(csv_input.shape[0])
##    print "randArray.shape",randArray.shape
    csv_input['rand'] = randArray
    csv_input.sort_values(by = 'rand',axis = 0,inplace = True )#��ֵ�����Ϊ��ȫ
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
    #����ѡȡ���������
    X = data.drop('label',axis = 1)
    X = X.drop('SampleDate',axis = 1)
    X = X.drop('picktime',axis = 1)
    X = X.drop('classes',axis = 1)
    X = X.drop('temp_label',axis = 1)    
    y = data.label
    print "data.label :y",y.shape

    ##ɸѡ������Χ��
    for wave in X.columns:
##        print wave
        if float(wave) <= params['wlengths'] or float(wave) >= params['wlengthe']:
            X = X.drop(wave,axis = 1)

    
    ##ȥ���������
    X,y = X[y != params['droplabel']],y[y != params['droplabel']]
##    print "X,y",X_new.shape,y.shape
    lambdstr = "lambda " + params['lambdastr']
    print "lambdstr",lambdstr
    X = X.apply(eval(lambdstr))
        
    ##lambda ���ʽ��ţ���԰� ȡָ���任�������뿴��
    ##ѡ�����ǰģ����ͬ������ѡ���������������ȡ
    print "test_params :" , params['selectfeature_params']
    selectmodel = SelectKBest(score_func = params['selectfeature_params']['score_func'], k = params['selectfeature_params']['k'])
    X_new = selectmodel.fit_transform(X, y)

    ##���Ǹ��ݲ��������������������ȡ��������      
##    X_new = SelectKBest(chi2,k = dimention).fit_transform(X, y)
##    print "X_new ***********",X_new.shape

    ##�淶����������ֵ
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
    #����ѡȡ���������
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
        X = X.diff(-1)   ##ǰ����+1�ͺ�����-1
        X = X.dropna().T  ##����dropna()��������ȥ��ȱʡ�����ݣ���Ϊ�����������һ�л�һ��ȱʡ����
    ##    print "dropna",X
        
        '''
        ��һ�����Ƕ����ݽ���Ԥ����ļ��ַ���������lambda���ʽ��һЩ������Ŀ�Ŀ��Ƿ��ܼ�ǿ��͹�Գ����ݵ�����
        '''
        X = X.apply(lambda x:x-x.min())  #�����������Ƶ���0��ʼ  
    ##    X = minmax_scale(X)
        print "Scale shape",X.shape
        temp = columns[:-1]
    ##    print "temp",temp
        X = pd.DataFrame(X,columns = columns[:-1])  ##���¹���һ��dataframe������X ��column���µ�columns
    ##    X = np.r_[columns[:-1] , X]  ����������ܲ��������е�
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
        X = X.apply(eval(lambdastr)) ##�����eval�����ܰ��ַ���ת�ɱ��ʽ
##        Xnew=Xnew.apply(lambda x: 10**x)   ##lambda ���ʽ��ţ���԰� ȡָ���任�������뿴����
##        Xnew = Xnew.apppy(lambda x:x)
        print "Applied Data",X
    ##����ɸѡ���������飬���������ģʽ
    print "F-mode :",preproAll.getFeaMode()
    chi2
    model = SelectKBest(chi2, k = preproAll.getFeaDimen())
    X_new_s = model.fit_transform(X, y)
    scores= model.scores_

    if preproAll.getFeaMode() == 'ANOVA':  ##ANOVA  f_classif: ANOVA F-value between label/feature for classification tasks
        model = SelectKBest(f_classif, k = preproAll.getFeaDimen)
        X_new_s = model.fit_transform(Xnew, y)
        scores= model.scores_

    ##�־û��õ�������Ϣ
    preproAll.setFeaParams(model.get_params(deep = True))
    print "trained Feaparams:",preproAll.getFeaParams()    

    ##�־û������÷�
    ScoreOfFea = pd.DataFrame(X.columns)
    ScoreOfFea['scores'] = scores.T
    ScoreOfFea.to_csv("D:\\Python\\result\\featurescore.csv", index = False, mode = 'wb+')
    ##numpy.savetxt("D:\\Python\\result\\featurescore.txt",scores.T)

 
    Data_new = np.c_[X_new_s , y]  ##ע����󷵻ص�����Ϻ�����壬ע���Ⱥ�˳�����Կ�����ԭʼ���ݵ�һ���ϼ������������У�����˵��ǩrow��ǰ�������������ں�
    print "########****Data_new",Data_new
    
    train = Data_new[:int(len(Data_new)*0.7)]  
    test = Data_new[int(len(Data_new)*0.7):]

    train_y = train[:,-1:]  
    train_x = train[:,:-1]

    test_y = test[:,-1:]  
    test_x = test[:,:-1]

    ##�������ݷ��������е����ݿ������׼ȷ�ȣ��е���Ȼ
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


def read_Kf(data_file,droplabel):
    import pandas as pd
    data = pd.read_csv(data_file)
    X = data.drop('label',axis = 1)
    y = data.label
    kf = KFold(len(y),n_folds = 3,shuffle = True )##kf ������index��������
    for train_index , test_index in kf:
        print "train,test " , train, test
    

def AddLabels(inputfile,labelArray,labelName):
    import pandas as pd
    csv_input = pd.read_csv(inputfile,error_bad_lines=False, skip_blank_lines=True ) 
    try:
        csv_input[labelName] = labelArray
    except Exception ,e:
        print Exception,":" ,e
    finally:
        csv_input.to_csv(inputfile, index = False, index_label = labelName, mode = 'wb+')
