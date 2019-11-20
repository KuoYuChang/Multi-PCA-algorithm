## using fast abod and lof here

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import roc_curve

from tqdm import tqdm

from pyod.models.abod import ABOD
from pyod.models.lof import LOF

## even with check_array in pyod, we transfer to numpy array ourselves

def fast_abod_pyod_once(X_nor, X_test, y_test, 
                        n_neighbors, contamination = 0.05):
    
    fastABOD = ABOD(n_neighbors=n_neighbors, contamination=contamination, method='fast')
    
    X_train = X_nor.astype(float).values.copy()
    
    fastABOD.fit(X_train)
    ## now threshold is determined
    
    y_pred = fastABOD.predict(X_test)
    scoreTable = fastABOD.decision_function(X_test)
    #print(scoreTable)
    scoreTable = np.nan_to_num(scoreTable, copy=True)
    
    ## confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)
    #tprW[trail] = tpr
    #fprW[trail] = fpr
    tprW = tpr
    fprW = fpr
    
    
    # Auc score
    auc = roc_auc_score(y_test,scoreTable)
    
    #print(tpr, fpr)
    #print(auc)
    
    return tprW, fprW, auc, scoreTable

def fast_abod_pyod_auc(X_nor, X_test, y_test, 
                        n_neighbors, contamination = 0.05):
    
    fastABOD = ABOD(n_neighbors=n_neighbors, contamination=contamination, method='fast')
    
    X_train = X_nor.astype(float).values.copy()
    
    fastABOD.fit(X_train)
    ## now threshold is determined
    
    #y_pred = fastABOD.predict(X_test)
    scoreTable = fastABOD.decision_function(X_test)
    #print(scoreTable)
    scoreTable = np.nan_to_num(scoreTable, copy=True)
    
    ## confusion matrix
    #tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    #tpr = tp/(tp+fn)
    #fpr = fp/(tn+fp)
    #tprW[trail] = tpr
    #fprW[trail] = fpr
    #tprW = tpr
    #fprW = fpr
    
    
    # Auc score
    auc = roc_auc_score(y_test,scoreTable)
    
    #print(tpr, fpr)
    #print(auc)
    
    return auc

def lof_pyod_once(X_nor, X_test, y_test, 
                  n_neighbors, contamination = 0.05):
    
    lof = LOF(n_neighbors=n_neighbors, contamination=contamination)
    
    X_train = X_nor.astype(float).values.copy()
    
    lof.fit(X_train)
    ## now threshold is determined
    
    y_pred = lof.predict(X_test)
    scoreTable = lof.decision_function(X_test)
    #print(scoreTable)
    scoreTable = np.nan_to_num(scoreTable, copy=True)
    
    ## confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)
    #tprW[trail] = tpr
    #fprW[trail] = fpr
    tprW = tpr
    fprW = fpr
    
    
    # Auc score
    auc = roc_auc_score(y_test,scoreTable)
    
    #print(tpr, fpr)
    #print(auc)
    
    return tprW, fprW, auc, scoreTable

def lof_pyod_auc(X_nor, X_test, y_test, 
                  n_neighbors, contamination = 0.05):
    
    lof = LOF(n_neighbors=n_neighbors, contamination=contamination)
    
    X_train = X_nor.astype(float).values.copy()
    
    lof.fit(X_train)
    ## now threshold is determined
    
    #y_pred = lof.predict(X_test)
    scoreTable = lof.decision_function(X_test)
    #print(scoreTable)
    scoreTable = np.nan_to_num(scoreTable, copy=True)
    
    ## confusion matrix
    #tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    #tpr = tp/(tp+fn)
    #fpr = fp/(tn+fp)
    #tprW[trail] = tpr
    #fprW[trail] = fpr
    #tprW = tpr
    #fprW = fpr
    
    
    # Auc score
    auc = roc_auc_score(y_test,scoreTable)
    
    #print(tpr, fpr)
    #print(auc)
    
    return auc

def lof_Ntrail_auc(currentData, y_Series,
               NTRAIL=5):
    k_neigh = 10+np.arange(6)
    dropArr = 0.02*(np.arange(1)+1)
    
    print(k_neigh)
    print(dropArr)
    
    currentWhole = pd.concat([currentData, y_Series], axis=1)
    
    tprW = np.zeros([len(k_neigh),len(dropArr), NTRAIL])
    fprW = np.zeros([len(k_neigh),len(dropArr), NTRAIL])
    aucW = np.zeros([len(k_neigh),len(dropArr),NTRAIL])
    
    P = len(k_neigh)
    Q = len(dropArr)
    
    ## only for get number of normal
    X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
    X_nor_pre = X_nor_temp.sample(frac=0.5)
    norNum = len(X_nor_pre)
    
    
    for trail in range(0,NTRAIL):
        X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
        X_nor_pre = X_nor_temp.sample(frac=0.5)
        X_nor_test = X_nor_temp.drop(X_nor_pre.index,axis=0)

        X_ano_temp = currentWhole.loc[currentWhole['label']==1].copy()
        
        
        X_ano = X_ano_temp
        ##
        '''
        if RANDOMPICK == False:
            X_ano = X_ano_temp
        else:
            ## 0.1 of number of all normal dataset
            print('Random pick anomaly')
            X_ano = X_ano_temp.sample(n=int(len(X_nor_temp)*0.1))
        '''


        X_test_temp = pd.concat([X_nor_test,X_ano],axis=0)
        X_test_pre = X_test_temp.sample(frac=1)

        ## drop label from dataset
        X_nor = X_nor_pre.drop(['label'],axis=1)
        y_test = X_test_pre['label'].copy()
        X_test = X_test_pre.drop(['label'],axis=1)

        #just for test
        print(len(X_nor),len(X_test))
        
        
        
        for pw in tqdm(range(0,P)):
            n_neigh = k_neigh[pw]
            #betaW = 1/(sRateW*(norNum))

            for qw in range(0,Q):
                dropRateT_W = dropArr[qw]

                ## detecting
                auc = lof_pyod_auc(X_nor, X_test, y_test, 
                                   n_neighbors=n_neigh, contamination=dropRateT_W)
                #tprW[pw,qw,trail] = tpr
                #fprW[pw,qw,trail] = fpr
                aucW[pw,qw,trail] = auc
                
                ## not to draw roc curves here

    ## end of all for loop

    ## about 9hr

    #avgTpr = tprW.mean(axis=2)
    #avgFpr = fprW.mean(axis=2)
    avgAuc = aucW.mean(axis=2)
    
    return k_neigh, dropArr, avgAuc

def lof_Ntrail_tpr(currentData, y_Series, k_neigh,
               NTRAIL=5):
    #k_neigh = 10+np.arange(6)
    dropArr = 0.02*(np.arange(10)+1)
    
    print(k_neigh)
    print(dropArr)
    
    currentWhole = pd.concat([currentData, y_Series], axis=1)
    
    #P = len(k_neigh)
    P = 1
    Q = len(dropArr)
    
    tprW = np.zeros([P,Q, NTRAIL])
    fprW = np.zeros([P,Q, NTRAIL])
    aucW = np.zeros([P,Q, NTRAIL])
    
    
    
    ## only for get number of normal
    X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
    X_nor_pre = X_nor_temp.sample(frac=0.5)
    norNum = len(X_nor_pre)
    
    
    for trail in range(0,NTRAIL):
        X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
        X_nor_pre = X_nor_temp.sample(frac=0.5)
        X_nor_test = X_nor_temp.drop(X_nor_pre.index,axis=0)

        X_ano_temp = currentWhole.loc[currentWhole['label']==1].copy()
        
        
        X_ano = X_ano_temp
        ##
        '''
        if RANDOMPICK == False:
            X_ano = X_ano_temp
        else:
            ## 0.1 of number of all normal dataset
            print('Random pick anomaly')
            X_ano = X_ano_temp.sample(n=int(len(X_nor_temp)*0.1))
        '''


        X_test_temp = pd.concat([X_nor_test,X_ano],axis=0)
        X_test_pre = X_test_temp.sample(frac=1)

        ## drop label from dataset
        X_nor = X_nor_pre.drop(['label'],axis=1)
        y_test = X_test_pre['label'].copy()
        X_test = X_test_pre.drop(['label'],axis=1)

        #just for test
        print(len(X_nor),len(X_test))
        
        
        
        for pw in range(0,P):
            n_neigh = k_neigh
            #betaW = 1/(sRateW*(norNum))

            for qw in tqdm(range(0,Q)):
                dropRateT_W = dropArr[qw]

                ## detecting
                tpr, fpr, auc, scoreTable = lof_pyod_once(X_nor, X_test, y_test, 
                                   n_neighbors=n_neigh, contamination=dropRateT_W)
                tprW[pw,qw,trail] = tpr
                fprW[pw,qw,trail] = fpr
                #aucW[pw,qw,trail] = auc
                
                ## not to draw roc curves here

    ## end of all for loop

    ## about 9hr

    avgTpr = tprW.mean(axis=2)
    avgFpr = fprW.mean(axis=2)
    #avgAuc = aucW.mean(axis=2)
    
    return k_neigh, dropArr, avgTpr, avgFpr

def fast_abod_Ntrail_auc(currentData, y_Series,
                    NTRAIL=5):
    k_neigh = 10+np.arange(6)
    dropArr = 0.02*(np.arange(1)+1)
    
    print(k_neigh)
    print(dropArr)
    
    currentWhole = pd.concat([currentData, y_Series], axis=1)
    
    tprW = np.zeros([len(k_neigh),len(dropArr), NTRAIL])
    fprW = np.zeros([len(k_neigh),len(dropArr), NTRAIL])
    aucW = np.zeros([len(k_neigh),len(dropArr),NTRAIL])
    
    P = len(k_neigh)
    Q = len(dropArr)
    
    ## only for get number of normal
    X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
    X_nor_pre = X_nor_temp.sample(frac=0.5)
    norNum = len(X_nor_pre)
    
    
    for trail in range(0,NTRAIL):
        X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
        X_nor_pre = X_nor_temp.sample(frac=0.5)
        X_nor_test = X_nor_temp.drop(X_nor_pre.index,axis=0)

        X_ano_temp = currentWhole.loc[currentWhole['label']==1].copy()
        
        
        X_ano = X_ano_temp
        ##
        '''
        if RANDOMPICK == False:
            X_ano = X_ano_temp
        else:
            ## 0.1 of number of all normal dataset
            print('Random pick anomaly')
            X_ano = X_ano_temp.sample(n=int(len(X_nor_temp)*0.1))
        '''


        X_test_temp = pd.concat([X_nor_test,X_ano],axis=0)
        X_test_pre = X_test_temp.sample(frac=1)

        ## drop label from dataset
        X_nor = X_nor_pre.drop(['label'],axis=1)
        y_test = X_test_pre['label'].copy()
        X_test = X_test_pre.drop(['label'],axis=1)

        #just for test
        print(len(X_nor),len(X_test))
        
        
        
        for pw in tqdm(range(0,P)):
            n_neigh = k_neigh[pw]
            #betaW = 1/(sRateW*(norNum))

            for qw in range(0,Q):
                dropRateT_W = dropArr[qw]

                ## detecting
                auc= fast_abod_pyod_auc(X_nor, X_test, y_test, 
                                        n_neighbors=n_neigh, contamination=dropRateT_W)
                #tprW[pw,qw,trail] = tpr
                #fprW[pw,qw,trail] = fpr
                aucW[pw,qw,trail] = auc
                
                ## not to draw roc curves here

    ## end of all for loop

    ## about 9hr

    #avgTpr = tprW.mean(axis=2)
    #avgFpr = fprW.mean(axis=2)
    avgAuc = aucW.mean(axis=2)
    
    return k_neigh, dropArr, avgAuc


def fast_abod_Ntrail_tpr(currentData, y_Series, k_neigh,
                    NTRAIL=5):
    #k_neigh = 10+np.arange(1)
    dropArr = 0.02*(np.arange(10)+1)
    
    print(k_neigh)
    print(dropArr)
    
    currentWhole = pd.concat([currentData, y_Series], axis=1)
    
    #P = len(k_neigh)
    P = 1
    Q = len(dropArr)
    
    tprW = np.zeros([P,Q, NTRAIL])
    fprW = np.zeros([P,Q, NTRAIL])
    aucW = np.zeros([P,Q,NTRAIL])
    
    
    
    ## only for get number of normal
    X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
    X_nor_pre = X_nor_temp.sample(frac=0.5)
    norNum = len(X_nor_pre)
    
    
    for trail in range(0,NTRAIL):
        X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
        X_nor_pre = X_nor_temp.sample(frac=0.5)
        X_nor_test = X_nor_temp.drop(X_nor_pre.index,axis=0)

        X_ano_temp = currentWhole.loc[currentWhole['label']==1].copy()
        
        
        X_ano = X_ano_temp
        ##
        '''
        if RANDOMPICK == False:
            X_ano = X_ano_temp
        else:
            ## 0.1 of number of all normal dataset
            print('Random pick anomaly')
            X_ano = X_ano_temp.sample(n=int(len(X_nor_temp)*0.1))
        '''


        X_test_temp = pd.concat([X_nor_test,X_ano],axis=0)
        X_test_pre = X_test_temp.sample(frac=1)

        ## drop label from dataset
        X_nor = X_nor_pre.drop(['label'],axis=1)
        y_test = X_test_pre['label'].copy()
        X_test = X_test_pre.drop(['label'],axis=1)

        #just for test
        print(len(X_nor),len(X_test))
        
        
        
        for pw in range(0,P):
            n_neigh = k_neigh
            #betaW = 1/(sRateW*(norNum))

            for qw in tqdm(range(0,Q)):
                dropRateT_W = dropArr[qw]

                ## detecting
                tpr, fpr, auc, scoreTable = fast_abod_pyod_once(X_nor, X_test, y_test, 
                                        n_neighbors=n_neigh, contamination=dropRateT_W)
                tprW[pw,qw,trail] = tpr
                fprW[pw,qw,trail] = fpr
                aucW[pw,qw,trail] = auc
                
                ## not to draw roc curves here

    ## end of all for loop

    ## about 9hr

    avgAuc = aucW.mean(axis=2)
    avgTpr = tprW.mean(axis=2)
    avgFpr = fprW.mean(axis=2)
    #avgAuc = aucW.mean(axis=2)
    
    stdAuc = aucW.std(axis=2)
    stdTpr = tprW.std(axis=2)
    stdFpr = fprW.std(axis=2)

    # for finding baseline fast
    return k_neigh, dropArr, avgAuc, avgTpr, avgFpr, stdAuc, stdTpr, stdFpr
