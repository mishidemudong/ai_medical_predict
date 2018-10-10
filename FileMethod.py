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
    csv_input.sort_values(by = 'rand',axis = 0,inplace = True )#用值排序较为安全
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
    #特征选取后的新数据
    X = data.drop('label',axis = 1)
    X = X.drop('SampleDate',axis = 1)
    X = X.drop('picktime',axis = 1)
    X = X.drop('classes',axis = 1)
    X = X.drop('temp_label',axis = 1)    
    y = data.label
    print "data.label :y",y.shape

    ##筛选波长范围内
    for wave in X.columns:
##        print wave
        if float(wave) <= params['wlengths'] or float(wave) >= params['wlengthe']:
            X = X.drop(wave,axis = 1)

    
    ##去除特殊类别
    X,y = X[y != params['droplabel']],y[y != params['droplabel']]
##    print "X,y",X_new.shape,y.shape
    lambdstr = "lambda " + params['lambdastr']
    print "lambdstr",lambdstr
    X = X.apply(eval(lambdstr))
        
    ##lambda 表达式很牛×对吧 取指数变换，详情请看博
    ##选择跟当前模型相同的特征选择参数进行特征提取
    print "test_params :" , params['selectfeature_params']
    selectmodel = SelectKBest(score_func = params['selectfeature_params']['score_func'], k = params['selectfeature_params']['k'])
    X_new = selectmodel.fit_transform(X, y)

    ##或是根据测试数据自身进行特征提取，不建议      
##    X_new = SelectKBest(chi2,k = dimention).fit_transform(X, y)
##    print "X_new ***********",X_new.shape

    ##规范化特征属性值
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
    #特征选取后的新数据
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
        X = X.diff(-1)   ##前向差分+1和后向差分-1
        X = X.dropna().T  ##利用dropna()方法可以去除缺省项数据，因为差分运算会产生一列或一行缺省数据
    ##    print "dropna",X
        
        '''
        这一部分是对数据进行预处理的几种方法，运用lambda表达式或一些方法，目的看是否能加强和凸显出数据的性质
        '''
        X = X.apply(lambda x:x-x.min())  #把整体数据移到从0开始  
    ##    X = minmax_scale(X)
        print "Scale shape",X.shape
        temp = columns[:-1]
    ##    print "temp",temp
        X = pd.DataFrame(X,columns = columns[:-1])  ##重新构造一个dataframe数据是X ，column是新的columns
    ##    X = np.r_[columns[:-1] , X]  这个函数功能不是想象中的
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
        X = X.apply(eval(lambdastr)) ##神奇的eval函数能把字符串转成表达式
##        Xnew=Xnew.apply(lambda x: 10**x)   ##lambda 表达式很牛×对吧 取指数变换，详情请看博客
##        Xnew = Xnew.apppy(lambda x:x)
        print "Applied Data",X
    ##特征筛选：卡方检验，方差分析等模式
    print "F-mode :",preproAll.getFeaMode()
    chi2
    model = SelectKBest(chi2, k = preproAll.getFeaDimen())
    X_new_s = model.fit_transform(X, y)
    scores= model.scores_

    if preproAll.getFeaMode() == 'ANOVA':  ##ANOVA  f_classif: ANOVA F-value between label/feature for classification tasks
        model = SelectKBest(f_classif, k = preproAll.getFeaDimen)
        X_new_s = model.fit_transform(Xnew, y)
        scores= model.scores_

    ##持久化好的特征信息
    preproAll.setFeaParams(model.get_params(deep = True))
    print "trained Feaparams:",preproAll.getFeaParams()    

    ##持久化特征得分
    ScoreOfFea = pd.DataFrame(X.columns)
    ScoreOfFea['scores'] = scores.T
    ScoreOfFea.to_csv("D:\\Python\\result\\featurescore.csv", index = False, mode = 'wb+')
    ##numpy.savetxt("D:\\Python\\result\\featurescore.txt",scores.T)

 
    Data_new = np.c_[X_new_s , y]  ##注意最后返回的是组合后的整体，注意先后顺序。所以可以在原始数据第一行上加上属性描述行，就是说标签row在前，被增加数据在后
    print "########****Data_new",Data_new
    
    train = Data_new[:int(len(Data_new)*0.7)]  
    test = Data_new[int(len(Data_new)*0.7):]

    train_y = train[:,-1:]  
    train_x = train[:,:-1]

    test_y = test[:,-1:]  
    test_x = test[:,:-1]

    ##正则化数据方法，对有的数据可以提高准确度，有的则不然
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
    kf = KFold(len(y),n_folds = 3,shuffle = True )##kf 里存的是index不是数据
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
