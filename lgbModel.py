#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:01:01 2018

@author: ldk
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,GaussianNoise,Embedding,Flatten,LSTM,SimpleRNN
from tensorflow.keras.layers import Dropout,BatchNormalization
from tensorflow.keras.constraints import MaxNorm as maxnorm  
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adagrad,Adam
from tensorflow.keras import Input
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,confusion_matrix,accuracy_score,f1_score
from sklearn.externals import joblib
import matplotlib  as plt

'''
For the boosting model, bigrams, trigrams and 4-gram tokens were created from all tokenized features.

For each feature tuple of the token name, value, and time-delta, we algorithmically created (described
below) a set of binary decision rules that partitioned examples into two classes.

'''



def featureselect(oriDF):
    feaSL_DF = oriDF.loc[:,['total_contact_time_x',
                             'time_per_mon_x']]
    return feaSL_DF

def left_joinFunc(df1,df2,colname):
    return pd.merge(df1, df2,how='left', on=colname)# 

def DnnNet(fea_x,train_y):
    
    model = Sequential()
    model.add(Embedding(1024, 100,input_length=fea_x.shape[1],embeddings_regularizer=regularizers.l2(0.00045),name='embedding'))   
    print ("1layer Embedding shape",model.output_shape)
    
    model.add(Flatten())
    model.add(GaussianNoise(stddev=0.8,name='GN'))
    model.add(Dense(output_dim=1024, input_dim=1024, init='uniform',activation='elu', W_constraint=maxnorm(1),name='Dense1'))
    model.add(Dense(output_dim=1024, input_dim=1024, init='uniform',activation='elu', W_constraint=maxnorm(1),name='Dense1'))
    model.add(Dropout(0.5,name='Dropout'))

    model.add(Dense(output_dim=2, input_dim=1024, init='uniform',activation='sigmoid', W_constraint=maxnorm(1),name='predict'))    
    
    '''
    lr=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=None,
    decay=0.,
    amsgrad=False,
    '''
    opt1 = Adam(lr=0.002, decay=1e-6)
    opt2 = Adam(lr=0.001, decay=1e-6)
    opt3 = Adam(lr=0.0005, decay=1e-6)
    
    epoch_1 = 1200
    epoch_2 = 500
    eopch_3 = 200
    
    k = 2
    batch_size_1 = 32*k
    batch_size_2 = 32*k
    batch_size_3 = 26*k
    
    print ("step 1...."  )
    
    model.compile(loss='categorical_crossentropy', optimizer=opt1 ,metrics=['accuracy'])  ##loss:categorical_crossentropy  sparse_categorical_crossentropy,
#        model = self.GridsearchCv(model)
    model.fit(fea_x, train_y, nb_epoch=epoch_1, batch_size=batch_size_1,shuffle = True,verbose=2) #4训练模型
        
    print ("step 2....")

    model.compile(loss='categorical_crossentropy', optimizer=opt2 ,metrics=['accuracy'])
    model.fit(fea_x, train_y, nb_epoch=epoch_2, batch_size=batch_size_2,shuffle = True,verbose=2) #4训练模型

    print ("step 3....")

    model.compile(loss='categorical_crossentropy', optimizer=opt3 ,metrics=['accuracy'])
    model.fit(fea_x, train_y, nb_epoch=eopch_3, batch_size=batch_size_3,shuffle = True,verbose=2) #4训练模型
    
    return model


def lgbmodelfit(alg, dtrain, dtest, predictors, pkl_file,useTrainCV=False, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        lgb_param = alg.get_params()
        lgbtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values)
#        lgbtest = lgb.Dataset(dtest[predictors].values)
        cvresult = lgb.cv(lgb_param, lgbtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #建模
    alg.fit(dtrain[predictors], dtrain['target'],eval_metric='auc')
    
    fea_imp = pd.DataFrame(alg.feature_importances_, columns= ['feature_importance'])
    fea_imp['feature'] = predictors
    
    #对训练集预测
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]
    
    train_f1_score = f1_score(dtrain['target'], dtrain_predictions)
    test_f1_score = f1_score(dtest['target'], dtest_predictions)

    #输出模型的一些结果
    print("准确率 : %.4g" % accuracy_score(dtrain['target'].values, dtrain_predictions))
    print("AUC 得分 (训练集): %f" % roc_auc_score(dtrain['target'], dtrain_predprob))
    print("f1_score (训练集): %f" % train_f1_score)
    print(confusion_matrix(dtrain['target'], dtrain_predictions))
            
    #输出模型的一些结果
    print("准确率 : %.4g" % accuracy_score(dtest['target'].values, dtest_predictions))
    print("AUC 得分 (测试集): %f" % roc_auc_score(dtest['target'], dtest_predprob))
    print("f1_score (测试集): %f" % test_f1_score)
    
    print(confusion_matrix(dtest['target'], dtest_predictions))
    if test_f1_score>0.0:
        print('save_model %s'%pkl_file)
        joblib.dump(alg, pkl_file)
#    print list(alg.booster().feature_importance)
    feat_imp = pd.Series(alg.booster().feature_importance).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    return fea_imp

def createLgbModel():
    gbm = lgb.LGBMClassifier(
                            objective='binary',
                            max_depth=3,
                            num_leaves=50,
                            silent=False,
                            learning_rate=0.1,
                            n_estimators=100,
                            subsample =0.8,
                            colsample_bytree=0.8,
                            reg_alpha = 0.00001,# L1 
                            reg_lambda = 0.00001 #L2
                            )
    
    return gbm

def lgb_train():
    savedmodel_dir = './evaluation/model/lightgb.pkl'
    resultfile = './result/result.csv'
    trainOrpredict = "train" 
    test4eval = "./data/train/TestData.csv"
    
    testfile = "./data/feature/test/feature_all.csv"
    
    target='target'
    IDcol = 'CUST_NO'
    type1 = 'type'
    
    oridata = pd.read_csv()
    feaDF = pd.read_csv()
    oridata = left_joinFunc(oridata,feaDF,'CUST_NO').fillna(-999).drop_duplicates("CUST_NO")
    
    if trainOrpredict == "train":
        oriTraindata = featureselect(oridata)
        Train_data,Test_data = train_test_split(oriTraindata, test_size = 0.3,random_state=7)
        
        predictors = [x for x in Train_data.columns if x not in [target, IDcol, type1]]
        
        feature_importance = lgbmodelfit(gbm,Train_data,Test_data,predictors,savedmodel_dir)
        print(feature_importance)    
        
        Test_data.to_csv(test4eval,index = False, index_label = None,mode = 'wb+')
    
    elif trainOrpredict == "predict":
        predictDF = pd.read_csv(testfile)
        fiteddata = predictDF.drop("CUST_NO",axis=1)
        labledDF = pd.DataFrame(predictDF["CUST_NO"])
        
        slfeaLableDF = featureselect(fiteddata)
        
        clf = joblib.load(savedmodel_dir)
        predict_label = clf.predict(slfeaLableDF)
        print(predict_label[:5])
    
        predict_score = clf.predict_proba(slfeaLableDF)
        print(predict_score[:5,0])  

def ckpt2pb():
    with tf.Session() as sess: 
        #初始化变量 
        sess.run(tf.global_variables_initializer()) 
        #获取最新的checkpoint，其实就是解析了checkpoint文件 
        latest_ckpt = tf.train.latest_checkpoint("./checkpoint_dir") 
        #加载图 
        restore_saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta') 
        #恢复图，即将weights等参数加入图对应位置中 
        restore_saver.restore(sess, latest_ckpt) 
        #将图中的变量转为常量 
        output_graph_def = tf.graph_util.convert_variables_to_constants( sess, sess.graph_def , ["out"]) 
        #将新的图保存到"/pretrained/graph.pb"文件中 
        tf.train.write_graph(output_graph_def, 'pretrained', "graph.pb", as_text=False)



def Dnn_train(_):
    data_file = "/home/python/keras/example/train/multi_apple_union.csv"
    direct = "/home/python/DistributeDL/model/"
    shuffle = True
    one_hot = True
    FLAGS = tf.app.flags.FLAGS
    
    nirdata = dataset.readdata(data_file,shuffle,one_hot)
    length = nirdata.train.data.shape[1]
    print ("length",length)
#    train_x, train_y, test_x, test_y = readdata(data_file,shuffle)
#    keep_prob = tf.placeholder("float")
#    
#    print"onehotencoder start...."
#
#    onehottrain_y = onehotencode(train_y)
#    onehottest_y = onehotencode(test_y)
    
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    
    ##
    ##1#
#    config = tf.ConfigProto()
#    config.gpu_options.per_process_gpu_memory_fraction = 0.3    
        
    ##2@自适应根据需要分配内存
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index,
                             config = config)
    print("Cluster job: %s, task_index: %d, target: %s" % (FLAGS.job_name, FLAGS.task_index, server.target))
    
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        from keras import backend as K
        
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):                                                          
            # Build model ...
            from keras.objectives import categorical_crossentropy
            K.set_learning_phase(1)
#            K.set_learning_phase(1) 
            x = tf.placeholder(tf.float32, shape=(None, length))
            
            model = ANNmodel()
            
            y = model(x)
                
            y_ = tf.placeholder(tf.float32,shape=(None, 4))

            
            cross_entropy = tf.reduce_mean(categorical_crossentropy(y_, y))
#            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

#            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_, logits = y))   

            global_step = tf.Variable(0)

            train_op1 = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy, global_step=global_step)     
#            train_op = tf.train.AdagradOptimizer(0.5).minimize(cross_entropy, global_step=global_step)
            train_op2 = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy, global_step=global_step)   
            train_op3 = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy, global_step=global_step) 
            # Test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            
            
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
          
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        # Create a "Supervisor", which oversees the training process.
        
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),  ##is_chief deside whitch worker to be the host
                                 logdir="/opt/tensor",
                                 init_op = init_op,
                                 summary_op = summary_op,
                                 saver = saver,
                                 global_step = global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization and restoring from
        # a checkpoint.
        wait_sess = sv.prepare_or_wait_for_session(server.target)
        K.set_session(wait_sess)
        # Start queue runners for the input pipelines (if ang).
        sv.start_queue_runners(wait_sess)
            # Generate input data batch
        # Loop until the supervisor shuts down (or 2000 steps have completed).
        start_time = time.time()
        
        print ("should_stop",sv.should_stop())
        ##step1
        step = 0
        while not sv.should_stop() and step < 1000:  ##次数尽量多一些可以更好的感受多机并行
            batch_xs, batch_ys = nirdata.train.next_batch(40)
#            print "real batch_ys is:" ,batch_ys
#            K.set_learning_phase(1)
            
            _, step,predictions = wait_sess.run([train_op1, global_step, y ], feed_dict={x: batch_xs, y_: batch_ys})
#            predict = predict_classes(predictions)
#            print "train_predict",predict  
#            print "onehot_predict",predictions
            if step % 10 == 0:
                print("Minibatch accuracy: %.01f%%" % (Trainaccuracy(predictions, batch_ys)))
                print("Step %d in task %d" % (step, FLAGS.task_index))
        
#        ##step2        
                # Assigns ops to the local worker by default.


        while not sv.should_stop() and step < 1500:  ##次数尽量多一些可以更好的感受多机并行
            batch_xs, batch_ys = nirdata.train.next_batch(35)
#            print "real batch_ys is:" ,batch_ys
#            K.set_learning_phase(1)
            _, step,predictions = wait_sess.run([train_op2, global_step, y ], feed_dict={x: batch_xs, y_: batch_ys})
#            predict = predict_classes(predictions)
#            print "train_predict",predict  
#            print "onehot_predict",predictions
            if step % 10 == 0:
                print ("train_op2")
                print("Minibatch accuracy: %.01f%%" % (Trainaccuracy(predictions, batch_ys)))
                print("Step %d in task %d" % (step, FLAGS.task_index))
        print ("should_stop",sv.should_stop())

        ##step3
        while not sv.should_stop() and step < 2000:  ##notice step is growing up ,add 10000 to
            batch_xs, batch_ys = nirdata.train.next_batch(30)
#            print "real batch_ys is:" ,batch_ys
            
            _, step,predictions = wait_sess.run([train_op3, global_step, y ], feed_dict={x: batch_xs, y_: batch_ys})
#            predict = predict_classes(predictions)
#            print "train_predict",predict  
#            print "onehot_predict",predictions
            if step % 10 == 0:
                print ("train_op3")
                print("Minibatch accuracy: %.01f%%" % (Trainaccuracy(predictions, batch_ys)))
                print("Step %d in task %d" % (step, FLAGS.task_index))
            

        print("done.")

        costtime = time.time() - start_time
        
        print ('training took %fs!' % costtime)
        if FLAGS.task_index == 1:
            print("accuracy: %f" % wait_sess.run(accuracy, feed_dict={x: nirdata.test.data, y_: nirdata.test.labels}))
            ##save model
            saver.save(wait_sess,direct+"model.ckpt")
            ##roload model
#            saver.restore(wait_sess,direct+"model.ckpt")
            print ("model is saved!")
        wait_sess.close()
        

if __name__ == "__main__":  




