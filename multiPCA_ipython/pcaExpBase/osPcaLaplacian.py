import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

from .Spectral_Clustering_Tool import spectral_result
from .Spectral_Clustering_Tool import colReorder

from .pyod_my import fast_abod_pyod_once
from .pyod_my import lof_pyod_once

def Track_w(x, w, d, beta):
    y = x.dot(w)
    dNew = beta*d + y*y
    e = x- y*w
    wNew = w + (y/dNew)*e
    return wNew, dNew

#becareful shape of sim_pool


def OD_onlinePCA_m(A, beta):
    #here A is 2-dim array
    
    #row, col of A
    n, p = A.shape
    
    A_m = np.mean(A, axis = 0)
    d = 0.0001
    u = np.ones(p)
    
    for i in range(0, n):
        u ,d = Track_w(A[i,:]-A_m, u, d, 1)
    #end
    u = u/np.linalg.norm(u)
    
    sim_pool = np.zeros(n)
    ratio = 1/(n*beta)
    
    for i in range(0,n):
        temp_mu = (A_m + ratio*A[i,:])/ (1+ratio)
        x = A[i, :] - temp_mu
        
        w1, d1 = Track_w(x, u, d, beta)
        w1 = w1/ np.linalg.norm(w1)
        
        sim_pool[i] = abs(u.dot(w1) )
        
        #if i%10000 == 0:
            #print('iteration '+ str(i))
        
    #end

    suspicious_index = np.argsort(sim_pool)
    suspicious_score = 1-sim_pool
    
    return suspicious_index, suspicious_score, u, d

#becareful shape of sim_pool

#suspicious_index: turn order accending original index into rank array

def OD_onlinePCA_forget(A, beta, ini_For=1):
    #here A is 2-dim array
    
    #row, col of A
    n, p = A.shape
    
    A_m = np.mean(A, axis = 0)
    d = 0.0001
    u = np.ones(p)
    
    for i in range(0, n):
        u ,d = Track_w(A[i,:]-A_m, u, d, ini_For)
        #
        #print(u)
    #end
    u = u/np.linalg.norm(u)
    
    
    sim_pool = np.zeros(n)
    ratio = 1/(n*beta)
    
    for i in range(0,n):
        temp_mu = (A_m + ratio*A[i,:])/ (1+ratio)
        x = A[i, :] - temp_mu
        
        w1, d1 = Track_w(x, u, d, beta)
        w1 = w1/ np.linalg.norm(w1)
        
        sim_pool[i] = abs(u.dot(w1) )
        
        #if i%10000 == 0:
            #print('iteration '+ str(i))
        
    #end
    
    suspicious_index = np.argsort(sim_pool)
    suspicious_score = 1-sim_pool
    
    return suspicious_index, suspicious_score, u, d, A_m

#-----------------------------------------------------------------------------
## online

## cleaning by droping top t%, remain top as threshold, via OD_onlinePCA_m
# drop with whole data, and clusterd data(k results)
# show threshold

# construct function since need implements many times

def cleanPCA(A, sRate, beta, ini_For = 1, drop_rate_t = 0.05):
    suspicious_index, suspicious_score, u, d, A_m = OD_onlinePCA_forget(A, beta, ini_For)
    
    #
    u0 = np.copy(u)
    
    #drop top t data, suspicious_index descending order
    #print("A.shape[1] ", A.shape[1])
    dropN = int(A.shape[0]*drop_rate_t)
    
    #last from dropped
    #or remaining top
    threshold = suspicious_score[suspicious_index[dropN]]
    #print("Threshold: ", threshold)
    
    cleanIdx = suspicious_index[dropN:len(suspicious_index)]
    ## random permute after drop out, on index
    cleanIdxRan = np.random.permutation(cleanIdx)
    cleanA = A[cleanIdxRan, :]
    
    
    
    
    #print(dropN, cleanA.shape[0])
    
    #just check
    #print(suspicious_score[suspicious_index])
    
    #print threshold, before deleting top datas
    # threshold = suspicious_score[cleanIdx[0]]
    
    
    
    ## see scatter plot of score
    ## also check here
    #tempYTrue = np.zeros(len(cleanIdx))
    #suspicious_plot(suspicious_score,tempYTrue,threshold)
    
    
    ## how about not a new u?, even old u, still strange
    #get new u

    betaR = 1/(sRate*cleanA.shape[0])
    suspicious_index, suspicious_score, u, d, A_m = OD_onlinePCA_forget(cleanA, betaR, ini_For)

    # turn off after investigate
    #print('After cleaning')
    #tempYTrue = np.zeros(len(suspicious_index))
    #suspicious_plot(suspicious_score,tempYTrue,threshold)
    
    cleanCosSim = abs(u.dot(u0))
    #print('before/after clean, similarity: ',cleanCosSim)

    # return pca from clean data, return threshold, new beta
    return u, d, threshold, betaR, cleanCosSim, A_m

def detectScore(x, u, d, betaR):
    newU, newD = Track_w(x, u, d, betaR)
    
    newU = newU / np.linalg.norm(newU)
    score = 1 - abs(u.dot(newU))
    
    return score
    
#detecting

## test ready

# return 01 array, 1 as anomaly
def test_pca(testA, u, d, A_m, sRate, betaR, threshold, UPDATE = False, upFor = 1):
    N = testA.shape[0]
    result = np.zeros(N)
    
    # need modified
    # get from sRate and betaR
    currentN = int(1/(sRate*betaR))
    
    ##
    scoreTable = np.zeros(N)
    
    u0 = np.copy(u)
    d0 = np.copy(d)
    
    for i in range(0, N):
        temp_mu = (A_m + sRate*testA[i,:])/ (1+sRate)
        x = testA[i, :] - temp_mu
        #scoring
        score = detectScore(x, u0, d0, betaR)
        
        # for auc
        scoreTable[i] = score
        
        #print(score)
        
        #check threshold
        if score >= threshold:
            result[i] = 1
        else:
            # regarded it as normal
            # update if needed
            if UPDATE:
                newU, newD = Track_w(x, u0, d0, upFor)
                
                u0 = newU / np.linalg.norm(newU)
                d0 = newD
                
                # update mean
                A_m = (currentN/(currentN+1))*A_m + (1/(currentN+1))*testA[i,:]
                
                ## recompute betaR
                currentN = currentN+1
                betaR = 1/(sRate*currentN)
                
                
                
            
        
    #end for
    
    
    return result, scoreTable



## ---------------------------------------------------------------------------
## feature clustering with graph Laplacian


## multiple cleaning

def cleanPCA_mul(fineClus_pre, feaNumArr, k, A, sRate, beta, ini_For = 1, drop_rate_t = 0.05):
    
    uArr = [None]*k
    dArr = np.zeros(k)
    thresholdArr = np.zeros(k)
    cleanCosArr = np.zeros(k)
    A_m_Arr = [None]*k
    # betaR should be the same for all clusters
    
    # A already an array
    
    for clusKey in fineClus_pre:
        featureNum = feaNumArr[clusKey]
        tempA = A[:, featureNum]
        u, d, threshold, betaR, cleanCos, A_m = cleanPCA(tempA, sRate, beta, ini_For, drop_rate_t)
        
        uArr[clusKey] = u
        dArr[clusKey] = d
        thresholdArr[clusKey] = threshold
        cleanCosArr[clusKey] = cleanCos
        A_m_Arr[clusKey] = A_m
        
    #maybe return objects, only uArr faced problem
    return uArr, dArr, thresholdArr, betaR, cleanCosArr, A_m_Arr



## scoring strategies for multiple clusters

# input scoringTable, thresholdArray, strategy(categorial variable)
# output boolean, if alert or not


def judgeAlert(scoringArr, thresholdArr, strategy):
    anomaly = False
    
    MAX = 0
    AVG = 1
    ANY = 2
    
    if strategy == MAX:
        score = np.max(scoringArr)
        threshold = np.max(thresholdArr)
        if score >= threshold:
            anomaly = True
        else :
            anomaly = False
        
    elif strategy == AVG:
        score = np.mean(scoringArr)
        threshold = np.mean(thresholdArr)
        if score >= threshold:
            anomaly = True
        else :
            anomaly = False
        
    elif strategy == ANY:
        k = len(thresholdArr)
        #print("k: ", k)
        for i in range(0, k):
            if scoringArr[i] >= thresholdArr[i]:
                anomaly = True
                #print(i, scoringArr[i], thresholdArr[i])
        #end for i
    

    
    return anomaly

# for multiple clustering, deciding scoring strategies
# detecting

## test ready

# return 01 array, 1 as anomaly
def test_pca_mul(feaNumArr, k, strategy, testA, uArr, dArr, A_m_Arr, sRate, betaR, thresholdArr, UPDATE = False, upFor = 1):
    N = testA.shape[0]
    result = np.zeros(N)
    scoringTable = np.zeros([N,k])
    
    uArr0 = np.copy(uArr)
    dArr0 = np.copy(dArr)
    
    # need modified
    # get from sRate and betaR
    currentN = int(1/(sRate*betaR))
    
    for i in tqdm(range(0, N)):
        x = testA[i, :]
        
        scoringArr = np.zeros(k)
        xprArr = [None]*k
        for clusKey in range(0, k):
            #scoring
            featureNum = feaNumArr[clusKey]
            xprTemp = x[featureNum]
            A_m = A_m_Arr[clusKey]
            temp_mu = (A_m + sRate*xprTemp)/ (1+sRate)
            xpr = xprTemp - temp_mu
            xprArr[clusKey] = xpr
            
            u0 = uArr0[clusKey]
            d0 = dArr0[clusKey]
            
            score = detectScore(xpr, u0, d0, betaR)
            scoringArr[clusKey] = score
            
            #print(clusKey, score)            
        #end for clusKey
        
        scoringTable[i,:] = scoringArr
        # strategy variables, categorical
        anomaly = judgeAlert(scoringArr, thresholdArr, strategy)
        
        # check bool variables
        if anomaly:
            result[i] = 1
        else:
            # regarded it as normal
            # update if needed
            if UPDATE:
                # update each clusters
                for clusKey in range(0, k):
                    xpr = xprArr[clusKey]
                    u0 = uArr0[clusKey]
                    d0 = dArr0[clusKey]
                    newU, newD = Track_w(xpr, u0, d0, upFor)

                    u0 = newU / np.linalg.norm(newU)
                    d0 = newD
                    
                    uArr0[clusKey] = u0
                    dArr0[clusKey] = d0
                    
                    #update mean
                    
                    #update betaR
            
        
    #end for
    
    
    return result, scoringTable



## function for dropping bad cluster

def checkCluster(thresholdArr, LowBDD, fineClus_pre, UpBdd=1):
    fineClus = np.where((thresholdArr>=LowBDD) & (thresholdArr<UpBdd))[0]
    #print('fineClus_pre: ', fineClus_pre)
    #print('fineClus: ', fineClus)
    fineClus = np.intersect1d(fineClus, fineClus_pre)
    
    return fineClus

# for multiple clustering, deciding scoring strategies
# detecting

## test ready

# return 01 array, 1 as anomaly
def test_pca_mul_select(fineClus, feaNumArr, k, strategy, testA, 
                        uArr, dArr, A_m_Arr, sRate, betaR, thresholdArr, 
                        UPDATE=False, upFor=1):
    N = testA.shape[0]
    result = np.zeros(N)
    scoringTable = np.zeros([N,k])
    
    uArr0 = np.copy(uArr)
    dArr0 = np.copy(dArr)
    
    # need modified
    # get from sRate and betaR
    currentN = int(1/(sRate*betaR))
    
    for i in tqdm(range(0, N)):
        x = testA[i, :]
        
        scoringArr = np.zeros(k)
        # xprArr stores mean shifted x
        xprArr = [None]*k
        for clusKey in fineClus:
            #scoring
            featureNum = feaNumArr[clusKey]
            xprTemp = x[featureNum]
            #mean shift
            A_m = A_m_Arr[clusKey]
            temp_mu = (A_m + sRate*xprTemp)/ (1+sRate)
            xpr = xprTemp - temp_mu
            xprArr[clusKey] = xpr
            
            u0 = uArr0[clusKey]
            d0 = dArr0[clusKey]
            
            score = detectScore(xpr, u0, d0, betaR)
            scoringArr[clusKey] = score
            
            #print(clusKey, score)            
        #end for clusKey
        
        scoringTable[i,:] = scoringArr
        # strategy variables, categorical
        anomaly = judgeAlert(scoringArr[fineClus], thresholdArr[fineClus], strategy)
        
        # check bool variables
        if anomaly:
            result[i] = 1
        else:
            # regarded it as normal
            # update if needed
            if UPDATE:
                ## recompute to update betaR
                currentN = currentN+1
                betaR = 1/(sRate*currentN)
                
                # update each clusters
                for clusKey in fineClus:
                    xpr = xprArr[clusKey]
                    u0 = uArr0[clusKey]
                    d0 = dArr0[clusKey]
                    newU, newD = Track_w(xpr, u0, d0, upFor)

                    u0 = newU / np.linalg.norm(newU)
                    d0 = newD
                    
                    uArr0[clusKey] = u0
                    dArr0[clusKey] = d0
                    
                    #update mean
                    A_m = A_m_Arr[clusKey]
                    A_m = (currentN/(currentN+1))*A_m + (1/(currentN+1))*xpr
                    A_m_Arr[clusKey] = A_m
                
                    
            
        
    #end for
    
    
    return result, scoringTable[:,fineClus]


## for output one score
def oneScore(scoringTable, strategy):
    MAX = 0
    AVG = 1
    
    if strategy == MAX:
        return np.amax(scoringTable, axis=1)
    elif strategy == AVG:
        return np.mean(scoringTable, axis=1)
    else:
        print('Wrong strategy')
        return np.zeros(1)
    
def oneThreshold(thresholdArr, strategy):
    MAX = 0
    AVG = 1
    
    if strategy == MAX:
        return np.amax(thresholdArr)
    elif strategy == AVG:
        return np.mean(thresholdArr)
    else:
        print('Wrong strategy')
        return np.zeros(2)

## -------------------------------------------------------------------------------------------------------

## construct N random trails

## follow coding style
## return exp result
def oneRandomTrailWhole(X_nor, X_test, y_test,
                        sRateW, betaW, ini_For_W, dropRateT_W,
                       UPDATE=False, AUC_GET=True):
    '''
    One time random trail for whole data detection and clustered data detection
    
    Parameter
    ---------
    norNum : 
    
    currentWhole
    
    sRate : sampling rate, also for recompute beta after cleaning process
    
    beta : from sampling rate(sRate), for initial training
    
    ini_For : 
    
    dropRateT:
    
    
    
    Returns
    -------

    
    tprW
    
    fprW
    
 
    
    '''

    y_true = y_test

    ## whole
    A = X_nor.astype(float).values.copy()
    A.shape

    #sRate = 0.3
    #beta = 1/(sRate*len(X_nor))
    #dropRateT = 0.02

    ## add AUC
    ## put tqdm in pca
    A = X_nor.astype(float).values.copy()


    u, d, threshold, betaR_W, cleanCos, A_m = cleanPCA(A, sRateW, betaW, ini_For=ini_For_W, drop_rate_t=dropRateT_W)

    ## for check
    #print('threshold: ', threshold)
    #print('--------------------')



    #get y_labels
    testA = X_test.astype(float).values.copy()

    # detecting
    ## UPDATE default True
    #if UPDATE==False:
    #    print('Not update')
    result, scoreTable = test_pca(testA, u, d, A_m, sRateW, betaR_W, threshold, UPDATE=UPDATE)
    


    tn, fp, fn, tp = confusion_matrix(y_true, result).ravel()

    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)
    #tprW[trail] = tpr
    #fprW[trail] = fpr
    tprW = tpr
    fprW = fpr
    
    # ref: https://stats.stackexchange.com/questions/95209/how-can-i-interpret-sklearn-confusion-matrix
    #print('tpr: ', tpr, ', fpr: ', fpr)
    #pd.crosstab(y_true, result ,rownames=['True'], colnames=['Predicted'], margins=True)
    
    if AUC_GET:
        auc = roc_auc_score(y_true,scoreTable)
    else:
        auc = 0
    
    #for checking logic error
    #print('Auc: ', auc)
    #suspicious_plot(scoreTable, y_true, threshold)
    
    return tprW, fprW, auc, threshold, scoreTable



## build basis experiment fucntion
## using previous reference names
## half normal, other normal and all anomaly
## finish in one block

## follow coding style
## return exp result
def oneRandomTrailClus(X_nor, X_test, y_test,
                       sRateClus, betaClus, ini_For_Clus, dropRateT_Clus,
                       k, thresLowBDD, strategy,
                       thresUpBDD=1,
                       UPDATE=False, NOR_RAN=False, Name=' '):
    '''
    One time random trail for whole data detection and clustered data detection
    
    Parameter
    ---------
    norNum : 
    
    currentWhole
    
    sRate : sampling rate, also for recompute beta after cleaning process
    
    beta : from sampling rate(sRate), for initial training
    
    ini_For : 
    
    dropRateT:
    
    thresBDD: 
    
    strategy : 
    
    
    
    Returns
    -------
    fineClus
    
    tprW
    
    fprW
    
    tprClus
    
    fprClus
    
    
    '''

    y_true = y_test

    ## -----------------------------------------------------------------
    
    ## do Laplacian here to get featureCluster
    
    ## no need anymore since spectral_result will check
    ### check if any a feature that std==0
    #STD_DES = X_nor.describe().loc['std']
    #zeros = np.where(STD_DES == 0)[0]
    #print(zeros)
    #if len(zeros)>0:
    #    print('number: ', len(zeros))
    #    print('Some features std==0')
    #    X_nor = X_nor.drop(zeros,axis=1)
    #end if
    
    #weight_abs = abs(X_nor.corr().values)
    
    feaNumArr, labels, REORDER, colOrder = spectral_result(X_nor, k_clusters=k, Name=Name, NOR_RAN=NOR_RAN)
    
    ## check fine clusters here first, preserve variables >= 2
    length_OK = np.zeros(k)
    for j in range(0,k):
        if len(feaNumArr[j]) >= 2:
            length_OK[j] = 1
    #end for j
    fineClus_pre = np.where(length_OK==1)[0]
    
    if len(fineClus_pre) < k:
        print("Some of the clusters only contain one variable!")
    
    
    if REORDER:
        X_nor = colReorder(X_nor, REORDER=REORDER, colOrder=colOrder)
    
    ## Laplacian
    currentTrain = X_nor.astype(float)
    tempTrainA = currentTrain.values.copy()

    ## can't deal with A array 
    ## need new A since dimension may changed
    uArr, dArr, thresholdArr, betaR_Clus, cleanCosArr, A_m_Arr = cleanPCA_mul(fineClus_pre, feaNumArr, k, tempTrainA, sRateClus, betaClus,
                                                                              ini_For=ini_For_Clus, drop_rate_t=dropRateT_Clus)

    print(thresholdArr)
    #BDD = 0.001
    fineClus = checkCluster(thresholdArr, thresLowBDD, fineClus_pre, thresUpBDD)
    print('Fine Clusters: ', fineClus)
    #fineClusObj.append(fineClus)

    #deg = 15
    #rad = np.deg2rad(deg)
    #BDD = np.cos(rad)


    #MAX = 0
    #AVG = 1
    #ANY = 2
    #strategy = MAX

    ## reorderTrain, reorderTest ready
    ## using same sampleRate, beta
    ## using same clean drop rate

    ## fix some reference names
    ## control fpr to compare

    tempTest = X_test

    currentTest = tempTest

    testClusA = currentTest.astype(float).values.copy()

    ## Update default true
    if UPDATE==False:
        print('Not update')
    resultPack = test_pca_mul_select(fineClus, feaNumArr, k, strategy, testClusA, uArr, dArr, A_m_Arr, 
                                     sRateClus, betaR_Clus, thresholdArr,
                                    UPDATE=UPDATE)
    resultFineClus = resultPack[0]
    scoringTable = resultPack[1]
    
    scoreOneTable = oneScore(scoringTable, strategy)
    #
    #print(scoreOneTable.shape)

    tn, fp, fn, tp = confusion_matrix(y_true, resultFineClus).ravel()
    
    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)
    #tprClus[trail] = tpr
    #fprClus[trail] = fpr
    tprClus = tpr
    fprClus = fpr

    # ref: https://stats.stackexchange.com/questions/95209/how-can-i-interpret-sklearn-confusion-matrix
    print('tpr: ', tpr, ', fpr: ', fpr)
    pd.crosstab(y_true, resultFineClus, rownames=['True'], colnames=['Predicted'], margins=True)
    
    auc = roc_auc_score(y_true, scoreOneTable)
    aucClus = auc
    
    ##for checking logic error
    #print('Auc: ', auc)
    #threshold = oneThreshold(thresholdArr,strategy)
    #print('thresholdClus: ',threshold)
    #suspicious_plot(scoreOneTable, y_true, threshold)
    
    return fineClus, tprClus, fprClus, aucClus, thresholdArr, scoreOneTable, feaNumArr


## ----------------------------------------------------------------------------------------------------------
## experiments

## half normal as training
## other half normal and anomaly as test

def nTrailsWhole(currentData, y_Series,
                 NTRAIL=5, RANDOMPICK=False,
                 UPDATE=False,
                SCALE=False, SCTYPE=0):

    sampArr = 0.01 * (np.arange(31)+10)
    #dropArr = 0.0025 * (np.arange(81))
    #dropArr = 0.005*(np.arange(41))
    dropArr = 0.01*(np.arange(20)+1)

    print(sampArr)
    print(dropArr)

    
    currentWhole = pd.concat([currentData, y_Series], axis=1)

    
    

    tprW = np.zeros([len(sampArr),len(dropArr), NTRAIL])
    fprW = np.zeros([len(sampArr),len(dropArr), NTRAIL])
    aucW = np.zeros([len(sampArr),len(dropArr),NTRAIL])

    #tprClus_avg = np.zeros([len(sampArr),len(dropArr), NTRAIL])
    #fprClus_avg = np.zeros([len(sampArr),len(dropArr), NTRAIL])

    #tprClus_max = np.zeros([len(sampArr),len(dropArr), NTRAIL])
    #fprClus_max = np.zeros([len(sampArr),len(dropArr), NTRAIL])

    P = len(sampArr)
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


        ##
        if RANDOMPICK == False:
            X_ano = X_ano_temp
        else:
            ## 0.1 of number of all normal dataset
            print('Random pick anomaly')
            X_ano = X_ano_temp.sample(n=int(len(X_nor_temp)*0.1))


        X_test_temp = pd.concat([X_nor_test,X_ano],axis=0)
        X_test_pre = X_test_temp.sample(frac=1)

        ## drop label from dataset
        X_nor = X_nor_pre.drop(['label'],axis=1)
        y_test = X_test_pre['label'].copy()
        X_test = X_test_pre.drop(['label'],axis=1)

        #just for test
        print(len(X_nor),len(X_test))
        
        #scaling dataset
        if SCALE:
            if SCTYPE==1:
                X_nor_s, train_max, train_min = scaleData(X_nor, SCATYPE=1)
                X_test_s = scaleTo01(X_test, train_max, train_min)
            elif SCTYPE==2:
                X_nor_s, train_mean, train_std = scaleData(X_nor, SCATYPE=2)
                X_test_s = scaleToN01(X_test, train_mean, train_std)
        else:
            X_nor_s = X_nor
            X_test_s = X_test
            

        ini_For_W = 1
        for pw in range(0,P):
            sRateW = sampArr[pw]
            betaW = 1/(sRateW*(norNum))

            for qw in tqdm(range(0,Q)):
                dropRateT_W = dropArr[qw]

                ## detecting
                tpr, fpr, auc, threshold, scoreTable = oneRandomTrailWhole(X_nor_s, X_test_s, y_test,
                             sRateW, betaW, ini_For_W, dropRateT_W,
                             UPDATE=UPDATE)
                tprW[pw,qw,trail] = tpr
                fprW[pw,qw,trail] = fpr
                aucW[pw,qw,trail] = auc
                
                ## not to draw roc curves here

    ## end of all for loop

    ## about 9hr

    avgTpr = tprW.mean(axis=2)
    avgFpr = fprW.mean(axis=2)
    avgAuc = aucW.mean(axis=2)
    
    return sampArr, dropArr, avgTpr, avgFpr, avgAuc

#####################################
## add scaling function
# scaling function
def scaleData(X, SCATYPE=0):
    UNSCALE = 0
    SCALE01 = 1
    SCALEN01 = 2
    
    if SCATYPE == SCALE01:
        print('Scaling to 01')
        newX = (X-X.min()) / (X.max()-X.min())
        return newX, X.max(), X.min()
    elif SCATYPE == SCALEN01:
        print('Scaling to N01')
        newX = (X-X.mean()) / (X.std())
        return newX, X.mean(), X.std()
    else:
        print('No scaling')
        return X

# scaling for test data
def scaleTo01(X_test, train_max, train_min):
    new_test = (X_test-train_min) / (train_max-train_min)
    return new_test

def scaleToN01(X_test, train_mean, train_std):
    new_test = (X_test-train_mean) / (train_std)
    return new_test

def getTrainTest(currentData, y_Series, SCALE=False, SCTYPE=1):
    currentWhole = pd.concat([currentData, y_Series], axis=1)
    
    X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
    X_nor_pre = X_nor_temp.sample(frac=0.5)
    X_nor_test = X_nor_temp.drop(X_nor_pre.index,axis=0)

    X_ano = currentWhole.loc[currentWhole['label']==1].copy()
    
    X_test_temp = pd.concat([X_nor_test,X_ano],axis=0)
    X_test_pre = X_test_temp.sample(frac=1)

    ## drop label from dataset
    X_nor = X_nor_pre.drop(['label'],axis=1)
    y_test = X_test_pre['label'].copy()
    X_test = X_test_pre.drop(['label'],axis=1)
    
    if SCALE:
        if SCTYPE==1:
            X_nor_s, train_max, train_min = scaleData(X_nor, SCATYPE=1)
            X_test_s = scaleTo01(X_test, train_max, train_min)
        elif SCTYPE==2:
            X_nor_s, train_mean, train_std = scaleData(X_nor, SCATYPE=2)
            X_test_s = scaleToN01(X_test, train_mean, train_std)
    else:
        X_nor_s = X_nor
        X_test_s = X_test
        
    return X_nor_s, X_test_s, y_test
   

#need control if need scale dataset
def nTrail_Whole_Laplacian_full(currentData, y_Series,
                           sRateW, ini_For_W, dropRateT_W,
                           sRateClus, ini_For_Clus, dropRateT_Clus,
                           k, thresLowBDD, strategy,
                           n_abod, contami_abod,
                           n_lof, contami_lof,
                           Name,
                           NTRAIL=5,
                           thresUpBDD=1,
                           RANDOMPICK=False,
                           UPDATE=False, NOR_RAN=False,
                               SCALE=False, SCTYPE=0):
    
    currentWhole = pd.concat([currentData, y_Series], axis=1)
    
    tprWhole = np.zeros(NTRAIL)
    fprWhole = np.zeros(NTRAIL)
    aucWhole = np.zeros(NTRAIL)

    ## add up abod
    tprAbod = np.zeros(NTRAIL)
    fprAbod = np.zeros(NTRAIL)
    aucAbod = np.zeros(NTRAIL)
    
    # add up lof
    tprLof = np.zeros(NTRAIL)
    fprLof = np.zeros(NTRAIL)
    aucLof = np.zeros(NTRAIL)
    
    
    fineClusObj = []
    commonClus = []
    
    
    tprLap = np.zeros(NTRAIL)
    fprLap = np.zeros(NTRAIL)
    aucLap = np.zeros(NTRAIL)

    thresholdArrTable = np.zeros([NTRAIL,k])
    
    rocInfo = []
    cutPoint = np.zeros([NTRAIL,4])
    
    
    ##
    X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
    X_nor_pre = X_nor_temp.sample(frac=0.5)
    norNum = len(X_nor_pre)
    betaW = 1/(sRateW*(norNum))
    betaClus = 1/(sRateClus*(norNum))



    for trail in range(0,NTRAIL):
        X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
        X_nor_pre = X_nor_temp.sample(frac=0.5)
        X_nor_test = X_nor_temp.drop(X_nor_pre.index,axis=0)

        X_ano_temp = currentWhole.loc[currentWhole['label']==1].copy()


        ##
        if RANDOMPICK == False:
            X_ano = X_ano_temp
        else:
            ## 0.1 of number of all normal dataset
            print('Random pick anomaly')            
            X_ano = X_ano_temp.sample(n=int(len(X_nor_temp)*0.1))


        X_test_temp = pd.concat([X_nor_test,X_ano],axis=0)
        X_test_pre = X_test_temp.sample(frac=1)

        ## drop label from dataset
        X_nor = X_nor_pre.drop(['label'],axis=1)
        y_test = X_test_pre['label'].copy()
        X_test = X_test_pre.drop(['label'],axis=1)

        #just for test
        #print(len(X_nor),len(X_test))
        
        #prepare if need scaling
        # scaling test data in the end
        #scaling dataset
        if SCALE:
            if SCTYPE==1:
                X_nor_s, train_max, train_min = scaleData(X_nor, SCATYPE=1)
                X_test_s = scaleTo01(X_test, train_max, train_min)
            elif SCTYPE==2:
                X_nor_s, train_mean, train_std = scaleData(X_nor, SCATYPE=2)
                X_test_s = scaleToN01(X_test, train_mean, train_std)
        else:
            X_nor_s = X_nor
            X_test_s = X_test



        ## detecting
        tpr, fpr, auc, threshold, scoreTable = oneRandomTrailWhole(X_nor_s, X_test_s, y_test,
                     sRateW, betaW, ini_For_W, dropRateT_W,
                                           UPDATE=UPDATE)

        tprWhole[trail] = tpr
        fprWhole[trail] = fpr
        aucWhole[trail] = auc

        print('tpr: ', tpr, ', fpr: ', fpr)
        print('Auc whole: ', auc)
        print('----------------------------------')
        
        ## ---------------------------------------------------------------------

        # Laplacian

        fineClus, tprClus, fprClus, aucClus, thresholdArr, scoreOneTable, feaNumArr = oneRandomTrailClus(X_nor_s, X_test_s, y_test,
                                                                               sRateClus, betaClus, ini_For_Clus, dropRateT_Clus,
                                                                               k, thresLowBDD, strategy,
                                                                               thresUpBDD,
                                                                              UPDATE=UPDATE, NOR_RAN=NOR_RAN, Name=Name)
        
        if trail == 0:
            commonClus = feaNumArr
        else:
            print('Find intersection')
            
            # find intersection
            for idx in range(0,k):
                comLen = np.zeros(k)
                for pick in range(0,k):
                    comPart = np.intersect1d(commonClus[idx], feaNumArr[pick])
                    comLen[pick] = len(comPart)
                    print(comPart)
                print('ComLen')
                print(comLen)
                Midx = np.argmax(comLen)
                commonClus[idx] = np.intersect1d(commonClus[idx], feaNumArr[Midx])
                print('pick')
                print(commonClus[idx])
                print('------------------------')
            print('########################')
        
        thresholdArrTable[trail,:] = thresholdArr
        
        fineClusObj.append(fineClus)
        tprLap[trail] = tprClus
        fprLap[trail] = fprClus
        aucLap[trail] = aucClus
        
        print('Auc clusters: ', aucClus)
        


        ## add abod and lof after both pca finished
        tprW, fprW, auc, scoreTableLof = lof_pyod_once(X_nor, X_test, y_test, n_neighbors=n_lof, contamination=contami_lof)
        tprLof[trail] = tprW
        fprLof[trail] = fprW
        aucLof[trail] = auc
        print('Auc Lof: ',auc)
        
        tprW, fprW, auc, scoreTableAbod = fast_abod_pyod_once(X_nor, X_test, y_test, n_neighbors=n_abod, contamination=contami_abod)
        tprAbod[trail] = tprW
        fprAbod[trail] = fprW
        aucAbod[trail] = auc
        print('Auc Abod: ', auc)
        
        # save info for each trail
        
        rocInfo = np.zeros([5,len(y_test)])
        rocInfo[0,:] = y_test
        rocInfo[1,:] = scoreTable
        rocInfo[2,:] = scoreOneTable
        rocInfo[3,:] = scoreTableLof
        rocInfo[4,:] = scoreTableAbod
        saveRocInfo = pd.DataFrame(rocInfo)
        saveRocInfo.to_csv('plotInfo/'+Name+'_rocInfo_'+str(trail)+'.csv', index=False, header=None)
        
        
        cutPoint[trail,:] = np.array([tpr, tprClus, tprLof[trail], tprAbod[trail]])
        
        
        ROC_plot(y_test, scoreTable, tpr, scoreOneTable, tprClus, scoreTableLof, tprLof[trail], scoreTableAbod, tprAbod[trail])
        
        

    ## end of all for loop

    
    
    saveCutPoint = pd.DataFrame(cutPoint)
    saveCutPoint.to_csv('plotInfo/'+Name+'_cutPoint.csv', index=False, header=None)
    
    print('For whole dataset: ')
    print('Auc: ', aucWhole.mean(), '+-', aucWhole.std())
    print('Tpr: ', tprWhole.mean(), '+-', tprWhole.std())
    print('Fpr: ', fprWhole.mean(), '+-', fprWhole.std())
    print('For Laplacian: ')
    print('Average threshold: ')
    print(thresholdArrTable.mean(axis=0))
    print('threshold in order: ')
    thresDescend = np.sort(thresholdArrTable.mean(axis=0))
    print(thresDescend)
    print('Auc: ', aucLap.mean(), '+-', aucLap.std())
    print('Tpr: ', tprLap.mean(), '+-', tprLap.std())
    print('Fpr: ', fprLap.mean(), '+-', fprLap.std())
    print('Fine clusters: ')
    print(fineClusObj)
    
    print('For Lof: ')
    print('Auc: ', aucLof.mean(), '+-', aucLof.std())
    print('Tpr: ', tprLof.mean(), '+-', tprLof.std())
    print('Fpr: ', fprLof.mean(), '+-', fprLof.std())
    
    print('For Abod: ')
    print('Auc: ', aucAbod.mean(), '+-', aucAbod.std())
    print('Tpr: ', tprAbod.mean(), '+-', tprAbod.std())
    print('Fpr: ', fprAbod.mean(), '+-', fprAbod.std())
    
    #save auc, tpr, fpr
    # concate into 2-dim array
    
    # name+auc
    saveScore = []
    saveScore.append(aucWhole)
    saveScore.append(tprWhole)
    saveScore.append(fprWhole)
    saveScore.append(aucLap)
    saveScore.append(tprLap)
    saveScore.append(fprLap)
    saveScore.append(aucLof)
    saveScore.append(tprLof)
    saveScore.append(fprLof)
    saveScore.append(aucAbod)
    saveScore.append(tprAbod)
    saveScore.append(fprAbod)
    
    saveScore = pd.DataFrame(saveScore)
    
    saveScore.to_csv('plotInfo/'+Name+'_aucTprFpr.csv', index=False, header=None)
    
    
    saveCommonClus = pd.DataFrame(commonClus)
    saveCommonClus.to_csv('plotInfo/'+Name+'_commClus.csv', index=False, header=None)
    countNum = 0
    for idx in range(0,k):
        print(commonClus[idx])
        countNum = countNum + len(commonClus[idx])
    #end for
    print('Total ', countNum, 'elements')
    
    return aucWhole, tprWhole, fprWhole, aucLap, tprLap, fprLap, thresDescend


def quickAucTprFpr(Name):
    Score = pd.read_csv('plotInfo/'+Name+'_aucTprFpr.csv', header=None)
    #print(Score)
    ScoreArr = Score.values.copy()
    
    
    
    aucWhole = ScoreArr[0,:]
    tprWhole = ScoreArr[1,:]
    fprWhole = ScoreArr[2,:]
    print('For whole dataset: ')
    print('Auc: ', aucWhole.mean(), '+-', aucWhole.std())
    print('Tpr: ', tprWhole.mean(), '+-', tprWhole.std())
    print('Fpr: ', fprWhole.mean(), '+-', fprWhole.std())
    print('For Laplacian: ')
    #print('Average threshold: ')
    #print(thresholdArrTable.mean(axis=0))
    #print('threshold in order: ')
    #thresDescend = np.sort(thresholdArrTable.mean(axis=0))
    #print(thresDescend)
    aucLap = ScoreArr[3,:]
    tprLap = ScoreArr[4,:]
    fprLap = ScoreArr[5,:]
    print('Auc: ', aucLap.mean(), '+-', aucLap.std())
    print('Tpr: ', tprLap.mean(), '+-', tprLap.std())
    print('Fpr: ', fprLap.mean(), '+-', fprLap.std())
    #print('Fine clusters: ')
    #print(fineClusObj)
    
    
    aucLof = ScoreArr[6,:]
    tprLof = ScoreArr[7,:]
    fprLof = ScoreArr[8,:]
    print('For Lof: ')
    print('Auc: ', aucLof.mean(), '+-', aucLof.std())
    print('Tpr: ', tprLof.mean(), '+-', tprLof.std())
    print('Fpr: ', fprLof.mean(), '+-', fprLof.std())
    
    aucAbod = ScoreArr[9,:]
    tprAbod = ScoreArr[10,:]
    fprAbod = ScoreArr[11,:]
    print('For Abod: ')
    print('Auc: ', aucAbod.mean(), '+-', aucAbod.std())
    print('Tpr: ', tprAbod.mean(), '+-', tprAbod.std())
    print('Fpr: ', fprAbod.mean(), '+-', fprAbod.std())
##    

def readCommonClus(Name):
    tempClus = pd.read_csv('plotInfo/'+Name+'_commClus.csv', header=None)
    
    #print(tempClus)
    
    commonClus = []
    k = len(tempClus)
    
    for i in range(0,k):
        tempRow = tempClus.iloc[i]
        
        Clus = tempRow[~pd.isnull(tempRow)]
        ClusNum = Clus.values.copy()
        print(ClusNum)
        commonClus.append(ClusNum)
    #end for
    
    #print(commonClus)
    
    return commonClus


def quick_ROC_plot(Name, NTRAIL=5):
    tempCut = pd.read_csv('plotInfo/'+Name+'_cutPoint.csv', header=None)
    cutPoint = tempCut.values.copy()
    # call funciton
    for idx in range(0,NTRAIL):
        tempInfo = pd.read_csv('plotInfo/'+Name+'_rocInfo_'+str(idx)+'.csv', header=None)


        infoAll = tempInfo.values.copy()
        

        
        ROC_plot(infoAll[0,:], infoAll[1,:], cutPoint[idx,0], infoAll[2,:], cutPoint[idx,1], infoAll[3,:], cutPoint[idx,2], infoAll[4,:], cutPoint[idx,3])
##    

#preserve function
# also need to control if scale dataset

def nTrail_Whole_Laplacian(currentData, y_Series,
                           sRateW, ini_For_W, dropRateT_W,
                           sRateClus, ini_For_Clus, dropRateT_Clus,
                           k, thresLowBDD, strategy,
                           NTRAIL=5,
                           thresUpBDD=1,
                           RANDOMPICK=False,
                           UPDATE=False, NOR_RAN=False,
                          SCALE=False, SCTYPE=0):
    
    currentWhole = pd.concat([currentData, y_Series], axis=1)
    
    tprWhole = np.zeros(NTRAIL)
    fprWhole = np.zeros(NTRAIL)
    aucWhole = np.zeros(NTRAIL)

    fineClusObj = []
    commonClus = []
    
    tprLap = np.zeros(NTRAIL)
    fprLap = np.zeros(NTRAIL)
    aucLap = np.zeros(NTRAIL)

    thresholdArrTable = np.zeros([NTRAIL,k])
    
    
    ##
    X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
    X_nor_pre = X_nor_temp.sample(frac=0.5)
    norNum = len(X_nor_pre)
    betaW = 1/(sRateW*(norNum))
    betaClus = 1/(sRateClus*(norNum))



    for trail in range(0,NTRAIL):
        X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
        X_nor_pre = X_nor_temp.sample(frac=0.5)
        X_nor_test = X_nor_temp.drop(X_nor_pre.index,axis=0)

        X_ano_temp = currentWhole.loc[currentWhole['label']==1].copy()


        ##
        if RANDOMPICK == False:
            X_ano = X_ano_temp
        else:
            ## 0.1 of number of all normal dataset
            print('Random pick anomaly')            
            X_ano = X_ano_temp.sample(n=int(len(X_nor_temp)*0.1))


        X_test_temp = pd.concat([X_nor_test,X_ano],axis=0)
        X_test_pre = X_test_temp.sample(frac=1)

        ## drop label from dataset
        X_nor = X_nor_pre.drop(['label'],axis=1)
        y_test = X_test_pre['label'].copy()
        X_test = X_test_pre.drop(['label'],axis=1)

        #just for test
        #print(len(X_nor),len(X_test))

        
        #scaling dataset
        if SCALE:
            if SCTYPE==1:
                X_nor_s, train_max, train_min = scaleData(X_nor, SCATYPE=1)
                X_test_s = scaleTo01(X_test, train_max, train_min)
            elif SCTYPE==2:
                X_nor_s, train_mean, train_std = scaleData(X_nor, SCATYPE=2)
                X_test_s = scaleToN01(X_test, train_mean, train_std)
        else:
            X_nor_s = X_nor
            X_test_s = X_test


        ## detecting
        tpr, fpr, auc, threshold, scoreTable = oneRandomTrailWhole(X_nor_s, X_test_s, y_test,
                     sRateW, betaW, ini_For_W, dropRateT_W,
                                           UPDATE=UPDATE)

        tprWhole[trail] = tpr
        fprWhole[trail] = fpr
        aucWhole[trail] = auc

        print('tpr: ', tpr, ', fpr: ', fpr)
        print('Auc whole: ', auc)
        print('----------------------------------')
        
        ## ---------------------------------------------------------------------

        # Laplacian

        fineClus, tprClus, fprClus, aucClus, thresholdArr, scoreOneTable, feaNumArr = oneRandomTrailClus(X_nor_s, X_test_s, y_test,
                                                                               sRateClus, betaClus, ini_For_Clus, dropRateT_Clus,
                                                                               k, thresLowBDD, strategy,
                                                                               thresUpBDD,
                                                                              UPDATE=UPDATE, NOR_RAN=NOR_RAN)
        
        if trail == 0:
            commonClus = feaNumArr
        else:
            print('Find intersection')
            
            # find intersection
            for idx in range(0,k):
                comLen = np.zeros(k)
                for pick in range(0,k):
                    comPart = np.intersect1d(commonClus[idx], feaNumArr[pick])
                    comLen[pick] = len(comPart)
                    print(comPart)
                print('ComLen')
                print(comLen)
                Midx = np.argmax(comLen)
                commonClus[idx] = np.intersect1d(commonClus[idx], feaNumArr[Midx])
                print('pick')
                print(commonClus[idx])
                print('------------------------')
            print('########################')
                
        
        thresholdArrTable[trail,:] = thresholdArr
        
        fineClusObj.append(fineClus)
        tprLap[trail] = tprClus
        fprLap[trail] = fprClus
        aucLap[trail] = aucClus
        
        print('Auc clusters: ', aucClus)
        ROC_plot(y_test, scoreTable, tpr, scoreOneTable, tprClus)



    ## end of all for loop

    print('For whole dataset: ')
    print('Auc: ', aucWhole.mean(), '+-', aucWhole.std())
    print('Tpr: ', tprWhole.mean(), '+-', tprWhole.std())
    print('Fpr: ', fprWhole.mean(), '+-', fprWhole.std())
    print('For Laplacian: ')
    print('Average threshold: ')
    print(thresholdArrTable.mean(axis=0))
    print('threshold in order: ')
    thresDescend = np.sort(thresholdArrTable.mean(axis=0))
    print(thresDescend)
    print('Auc: ', aucLap.mean(), '+-', aucLap.std())
    print('Tpr: ', tprLap.mean(), '+-', tprLap.std())
    print('Fpr: ', fprLap.mean(), '+-', fprLap.std())
    print('Fine clusters: ')
    print(fineClusObj)
    print('Common clusters: ')
    
    ## put common clusters into pandas
    ## save in csv
    
    countNum = 0
    for idx in range(0,k):
        print(commonClus[idx])
        countNum = countNum + len(commonClus[idx])
    #end for
    print('Total ', countNum, 'elements')
                                  
    return aucWhole, tprWhole, fprWhole, aucLap, tprLap, fprLap, thresDescend


## suspicious score plot
## scatter and threshold line
def suspicious_plot(suspicious_score, y_true, threshold):
    print('Suspicious score plot')
    x_index = np.arange(len(y_true))
    norIdx = np.where(y_true==0)[0]
    anoIdx = np.where(y_true==1)[0]
    plt.scatter(x_index[norIdx], suspicious_score[norIdx],s=1)
    plt.scatter(x_index[anoIdx],suspicious_score[anoIdx],color='r',s=1)
    plt.axhline(y=threshold,color='orange')
    #optional
    plt.ylim(0,1)
    plt.show()
## end of suppicious_plot





# consider plot all methods
def ROC_plot(y_true, score_whole, tprW_L, score_clus, tprClu_L, 
             score_lof=np.zeros(1), tprLof=np.zeros(1), score_abod=np.zeros(1), tprAbod=np.zeros(1)):
    
    ## save informations of figures needed
    
    
    # temparately zoom in 
    plt.rcParams["figure.figsize"][0] = 15
    plt.rcParams["figure.figsize"][1] = 10
    
    fprC, tprC, thresC = roc_curve(y_true, score_clus)
    plt.plot(fprC, tprC, color='darkorange', lw=3, label='Multi-view Online osPCA')
    plt.axhline(y=tprClu_L,color='darkorange',lw=3)
    fpr, tpr, thres = roc_curve(y_true, score_whole)
    plt.plot(fpr, tpr, color='green', lw=3, linestyle='--', label='Online osPCA')
    plt.axhline(y=tprW_L,color='green', linestyle='--',lw=3)
    #plt.axhline(y=tprClu_L,color='darkorange', linestyle='--')
    
    
    
    # just simple check that lof and abod are not default
    if len(score_lof)>1:
        fpr, tpr, thres = roc_curve(y_true, score_lof)
        plt.plot(fpr, tpr, color='blue', lw=2, linestyle=':', label='LOF')
        plt.axhline(y=tprLof,color='blue', linestyle=':', lw=2)
        
        fpr, tpr, thres = roc_curve(y_true, score_abod)
        plt.plot(fpr, tpr, color='purple', lw=2, linestyle=(0, (3, 1, 1, 1)), label='ABOD')
        plt.axhline(y=tprAbod,color='purple', linestyle=(0, (3, 1, 1, 1)), lw=2)

    plt.legend(loc='lower right')
    
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    
    plt.show()

    plt.rcParams["figure.figsize"][0] = 9
    plt.rcParams["figure.figsize"][1] = 6


##




'''
def train_opti_paper(currentData, y_Series, UPDATE=False, NTRAIL=1):
    ## NTRAIL in cases
    
    sampArr = 0.01 * (np.arange(31)+10)
    #dropArr = 0.0025 * (np.arange(81))
    #dropArr = 0.005*(np.arange(41))
    dropArr = 0.01*(np.arange(20)+1)

    print(sampArr)
    print(dropArr)

    
    currentWhole = pd.concat([currentData, y_Series], axis=1)

    
    

    tprW = np.zeros([len(sampArr),len(dropArr), NTRAIL])
    fprW = np.zeros([len(sampArr),len(dropArr), NTRAIL])
    aucW = np.zeros([len(sampArr),len(dropArr),NTRAIL])
    meanScore = np.zeros([len(sampArr),len(dropArr),NTRAIL])
    stdScore = np.zeros([len(sampArr),len(dropArr),NTRAIL])
    #thresGapW = np.zeros([len(sampArr),len(dropArr),NTRAIL])
    remainMean= np.zeros([len(sampArr),len(dropArr),NTRAIL])
    remainStd= np.zeros([len(sampArr),len(dropArr),NTRAIL])
    
    # try only do one time
    #tprW = np.zeros([len(sampArr),len(dropArr)])
    #fprW = np.zeros([len(sampArr),len(dropArr)])
    #aucW = np.zeros([len(sampArr),len(dropArr)])
    #thresGapW = np.zeros([len(sampArr),len(dropArr)])
    
    
    P = len(sampArr)
    Q = len(dropArr)
    ini_For_W = 1
    
    ## using oneRandomWhole
    for trail in range(0,NTRAIL):
        #print('Select parameters')
        
        ## train, opti, test set
        X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
        ## using sample 2 times
        X_train_temp = X_nor_temp.sample(frac=1/3)

        X_nor_temp2 = X_nor_temp.drop(X_train_temp.index,axis=0)
        X_vali_temp = X_nor_temp2.sample(frac=0.5)#0.33... / 0.66... = 0.5

        X_nor_test = X_nor_temp2.drop(X_vali_temp.index,axis=0)
        X_ano = currentWhole.loc[currentWhole['label']==1].copy()

        #print(np.intersect1d(X_ano.index.values.copy(),X_nor_test.index.values.copy()))
        #print(len(X_train_temp), len(X_vali_temp), len(X_nor_test))

        X_test_temp = pd.concat([X_nor_test,X_ano],axis=0)
        X_test_temp2 = X_test_temp.sample(frac=1)
        y_vali = X_vali_temp['label'].copy()
        y_test = X_test_temp2['label'].copy()

        #print(X_train_temp.head())
        #print(y_test.value_counts())

        ## drop label at last
        #
        X_train= X_train_temp.drop(['label'],axis=1)
        X_vali = X_vali_temp.drop(['label'],axis=1)
        X_test = X_test_temp2.drop(['label'],axis=1)
        
        norNum = len(X_train)
        
        for pw in range(0,P):
            sRateW = sampArr[pw]
            betaW = 1/(sRateW*(norNum))

            for qw in tqdm(range(0,Q)):
                dropRateT_W = dropArr[qw]

                ## detecting
                tpr, fpr, auc, threshold, scoreTable = oneRandomTrailWhole(X_train, X_vali, y_vali,
                             sRateW, betaW, ini_For_W, dropRateT_W,
                             UPDATE=UPDATE, AUC_GET=False)
                tprW[pw,qw,trail] = tpr
                fprW[pw,qw,trail] = fpr
                #aucW[pw,qw,trail] = auc
                meanScore[pw,qw,trail] = np.mean(scoreTable)
                stdScore[pw,qw,trail] = np.std(scoreTable)
                
                remainMean[pw,qw,trail], remainStd[pw,qw,trail] = remainScoreThresRatio(scoreTable, dropRateT_W, threshold)
                
                #scoreMax = np.amax(scoreTable)
                #thresGapW[pw,qw,trail] = threshold-scoreMax #preserve sign
                
                

        
    
    ## end trail
    
    avgTpr = tprW.mean(axis=2)
    avgFpr = fprW.mean(axis=2)
    avgAuc = aucW.mean(axis=2)
    avgMeanSc = meanScore.mean(axis=2)
    avgStdSc = stdScore.mean(axis=2)
    avgReMean = remainMean.mean(axis=2)
    avgReStd = remainStd.mean(axis=2)
    #avgThresGap = thresGapW.mean(axis=2)
    
    ## testing, or return testing set, return seems better
    
    return sampArr, dropArr, avgTpr, avgFpr, avgAuc, avgReMean, avgReStd, X_test, y_test
        
### end nTrail_paper
'''


def trainOpti_chronologically(currentData, y_Series, UPDATE=False):
    print('Dataset ordered chronologically')
    
    ## NTRAIL in cases
    
    sampArr = 0.01 * (np.arange(31)+10)
    #dropArr = 0.0025 * (np.arange(81))
    #dropArr = 0.005*(np.arange(41))
    dropArr = 0.01*(np.arange(20)+1)

    print(sampArr)
    print(dropArr)

    
    currentWhole = pd.concat([currentData, y_Series], axis=1)

    
    
    ## do only in once
    tprW = np.zeros([len(sampArr),len(dropArr)])
    fprW = np.zeros([len(sampArr),len(dropArr)])
    aucW = np.zeros([len(sampArr),len(dropArr)])
    #meanScore = np.zeros([len(sampArr),len(dropArr)])
    #stdScore = np.zeros([len(sampArr),len(dropArr)])
    #thresGapW = np.zeros([len(sampArr),len(dropArr)])
    scoreMean= np.zeros([len(sampArr),len(dropArr)])
    scoreStd= np.zeros([len(sampArr),len(dropArr)])
    
    thresTable = np.zeros([len(sampArr),len(dropArr)])
    
    P = len(sampArr)
    Q = len(dropArr)
    ini_For_W = 1
    
    
    #print('Select parameters')
    
    ## not shuffle
    ## data sorted with time

    ## train, opti, test set
    X_nor_temp = currentWhole.loc[currentWhole['label']==0].copy()
    ## need directly split dataset
    #X_train_temp = X_nor_temp.sample(frac=1/3)
    norN = len(X_nor_temp)
    X_train_temp = X_nor_temp.iloc[0:int(norN/3),:].copy()
    X_vali_temp = X_nor_temp.iloc[int(norN/3)+1:int(2*norN/3),:].copy()
    X_nor_test = X_nor_temp.iloc[int(2*norN/3)+1:norN,:].copy()

    #X_nor_temp2 = X_nor_temp.drop(X_train_temp.index,axis=0)
    #X_vali_temp = X_nor_temp2.sample(frac=0.5)#0.33... / 0.66... = 0.5

    #X_nor_test = X_nor_temp2.drop(X_vali_temp.index,axis=0)
    X_ano = currentWhole.loc[currentWhole['label']==1].copy()

    #print(np.intersect1d(X_ano.index.values.copy(),X_nor_test.index.values.copy()))
    #print(len(X_train_temp), len(X_vali_temp), len(X_nor_test))

    X_test_temp = pd.concat([X_nor_test,X_ano],axis=0)
    X_test_temp2 = X_test_temp.sample(frac=1)
    y_vali = X_vali_temp['label'].copy()
    y_test = X_test_temp2['label'].copy()

    #print(X_train_temp.head())
    #print(y_test.value_counts())

    ## drop label at last
    #
    X_train= X_train_temp.drop(['label'],axis=1)
    X_vali = X_vali_temp.drop(['label'],axis=1)
    X_test = X_test_temp2.drop(['label'],axis=1)

    norNum = len(X_train)

    for pw in range(0,P):
        sRateW = sampArr[pw]
        betaW = 1/(sRateW*(norNum))

        for qw in tqdm(range(0,Q)):
            dropRateT_W = dropArr[qw]

            ## detecting
            tpr, fpr, auc, threshold, scoreTable = oneRandomTrailWhole(X_train, X_vali, y_vali,
                                                                       sRateW, betaW, ini_For_W, dropRateT_W,
                                                                       UPDATE=UPDATE, AUC_GET=False)
            tprW[pw,qw] = tpr
            fprW[pw,qw] = fpr
            #aucW[pw,qw,trail] = auc
            #meanScore[pw,qw] = np.mean(scoreTable)
            #stdScore[pw,qw] = np.std(scoreTable)
            
            #print(np.mean(scoreTable), np.std(scoreTable))
            
            #print(-np.sort(-scoreTable))
            
            ## directly using whole score/threshold
            scoreMean[pw,qw] = np.mean(scoreTable)
            scoreStd[pw,qw] = np.std(scoreTable)
            
            thresTable[pw,qw] = threshold
            
            #remainMean[pw,qw], remainStd[pw,qw] = remainScoreThresRatio(scoreTable, dropRateT_W, threshold)
            #print(remainMean[pw,qw], remainStd[pq,qw])

            #scoreMax = np.amax(scoreTable)
            #thresGapW[pw,qw,trail] = threshold-scoreMax #preserve sign
    
    
    
    
    #return test set if necessary
    return sampArr, dropArr, tprW, fprW, scoreMean, scoreStd, thresTable, X_train, X_vali, y_vali, X_test, y_test

def findLeastScore(scoreTable, pick=1, ASCEND=True):
    ## using for everything
    if ASCEND:
        tempSort = np.argsort(scoreTable,axis=None)
    else:
        tempSort = np.argsort(-scoreTable,axis=None)
    tempBest = tempSort[0:pick]
    
    bestIdx = np.unravel_index(tempBest, scoreTable.shape)
    print(scoreTable[bestIdx])
    bestIdx = np.transpose(np.array(bestIdx))
    
    return bestIdx


def fprCleanRatioBest(avgFpr, cleanArr, pick=1, basis=1):
    ratio = np.divide(avgFpr,cleanArr)-1
    absRatio = np.absolute(ratio)
    
    #print(np.argsort(absRatio,axis=None))
    
    ## take part of it
    tempArgsort = np.argsort(absRatio,axis=None)
    #print(tempArgSort)
    
    tempBest = tempArgsort[0:pick]
    
    bestIdx = np.unravel_index(tempBest, absRatio.shape)
    print(ratio[bestIdx])
    bestIdx = np.transpose(np.array(bestIdx))
    
    
    
    return bestIdx
    
    

def remainScoreThresRatio(scoreTable, dropRateT_W, threshold):
    ## clean validation set
    ## scoreTable 1d array, check scoreOneTable
    dropN = int(len(scoreTable)*dropRateT_W)

    #from big to small
    suspicious_index = np.argsort(-scoreTable)

    cleanIdx = suspicious_index[dropN:len(suspicious_index)]
    remainScore = scoreTable[cleanIdx]
    
    remainMean = (np.mean(remainScore))
    remainStd = (np.std(remainScore))
    print(len(remainScore), remainScore)
    
    print(remainMean, remainStd)
    
    return remainMean, remainStd


def trainValiWhole_paper(X_nor, X_test, y_test,
                        sRateW, betaW, ini_For_W, dropRateT_W,
                       UPDATE=True):
    '''
    One time random trail for whole data detection and clustered data detection
    
    Parameter
    ---------
    norNum : 
    
    currentWhole
    
    sRate : sampling rate, also for recompute beta after cleaning process
    
    beta : from sampling rate(sRate), for initial training
    
    ini_For : 
    
    dropRateT:
    
    
    
    Returns
    -------

    
    tprW
    
    fprW
    
 
    
    '''

    y_true = y_test

    ## whole
    A = X_nor.astype(float).values.copy()
    A.shape

    #sRate = 0.3
    #beta = 1/(sRate*len(X_nor))
    #dropRateT = 0.02

    ## add AUC
    ## put tqdm in pca
    A = X_nor.astype(float).values.copy()


    u, d, threshold, betaR_W, cleanCos, A_m = cleanPCA(A, sRateW, betaW, ini_For=ini_For_W, drop_rate_t=dropRateT_W)

    ## for check
    #print('threshold: ', threshold)
    #print('--------------------')
    
    # preserve
    u0 = np.copy(u)
    d0 = d



    #get y_labels
    testA = X_test.astype(float).values.copy()

    # detecting
    ## UPDATE default True
    #if UPDATE==False:
    #    print('Not update')
    result, scoreTable = test_pca(testA, u, d, A_m, sRateW, betaR_W, threshold, UPDATE=UPDATE)
    


    tn, fp, fn, tp = confusion_matrix(y_true, result).ravel()

    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)
    #tprW[trail] = tpr
    #fprW[trail] = fpr
    tprW = tpr
    fprW = fpr
    
    # ref: https://stats.stackexchange.com/questions/95209/how-can-i-interpret-sklearn-confusion-matrix
    #print('tpr: ', tpr, ', fpr: ', fpr)
    #pd.crosstab(y_true, result ,rownames=['True'], colnames=['Predicted'], margins=True)
    
    #if AUC_GET:
    #    auc = roc_auc_score(y_true,scoreTable)
    #else:
    #    auc = 0
    
    #for checking logic error
    #print('Auc: ', auc)
    #suspicious_plot(scoreTable, y_true, threshold)
    
    return tprW, fprW, threshold, scoreTable, betaR_W, u0, d0, A_m

def trainValiClus(X_nor, X_test, y_test,
                       sRateClus, betaClus, ini_For_Clus, dropRateT_Clus,
                       k, thresLowBDD, strategy,
                       thresUpBDD=1,
                       UPDATE=True):
    '''
    One time random trail for whole data detection and clustered data detection
    
    Parameter
    ---------
    norNum : 
    
    currentWhole
    
    sRate : sampling rate, also for recompute beta after cleaning process
    
    beta : from sampling rate(sRate), for initial training
    
    ini_For : 
    
    dropRateT:
    
    thresBDD: 
    
    strategy : 
    
    
    
    Returns
    -------
    fineClus
    
    tprW
    
    fprW
    
    tprClus
    
    fprClus
    
    
    '''

    y_true = y_test

    ## -----------------------------------------------------------------
    
    ## do Laplacian here to get featureCluster
    
    ## no need anymore since spectral_result will check
    ### check if any a feature that std==0
    #STD_DES = X_nor.describe().loc['std']
    #zeros = np.where(STD_DES == 0)[0]
    #print(zeros)
    #if len(zeros)>0:
    #    print('number: ', len(zeros))
    #    print('Some features std==0')
    #    X_nor = X_nor.drop(zeros,axis=1)
    #end if
    
    #weight_abs = abs(X_nor.corr().values)
    
    feaNumArr, labels, REORDER, colOrder = spectral_result(X_nor, k_clusters=k)
    
    if REORDER:
        X_nor = colReorder(X_nor, REORDER=REORDER, colOrder=colOrder)
    
    ## Laplacian
    currentTrain = X_nor.astype(float)
    tempTrainA = currentTrain.values.copy()

    ## can't deal with A array 
    ## need new A since dimension may changed
    uArr, dArr, thresholdArr, betaR_Clus, cleanCosArr, A_m_Arr = cleanPCA_mul(feaNumArr, k, tempTrainA, sRateClus, betaClus,
                                                                              ini_For=ini_For_Clus, drop_rate_t=dropRateT_Clus)

    print(thresholdArr)
    #BDD = 0.001
    fineClus = checkCluster(thresholdArr, thresLowBDD, thresUpBDD)
    print('Fine Clusters: ', fineClus)
    #fineClusObj.append(fineClus)

    #deg = 15
    #rad = np.deg2rad(deg)
    #BDD = np.cos(rad)


    #MAX = 0
    #AVG = 1
    #ANY = 2
    #strategy = MAX

    ## reorderTrain, reorderTest ready
    ## using same sampleRate, beta
    ## using same clean drop rate

    ## fix some reference names
    ## control fpr to compare

    tempTest = X_test

    currentTest = tempTest

    testClusA = currentTest.astype(float).values.copy()

    ## Update default true
    if UPDATE==False:
        print('Not update')
    resultPack = test_pca_mul_select(fineClus, feaNumArr, k, strategy, testClusA, uArr, dArr, A_m_Arr, 
                                     sRateClus, betaR_Clus, thresholdArr,
                                    UPDATE=UPDATE)
    resultFineClus = resultPack[0]
    scoringTable = resultPack[1]
    
    scoreOneTable = oneScore(scoringTable, strategy)
    #
    #print(scoreOneTable.shape)

    tn, fp, fn, tp = confusion_matrix(y_true, resultFineClus).ravel()
    
    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)
    #tprClus[trail] = tpr
    #fprClus[trail] = fpr
    tprClus = tpr
    fprClus = fpr

    # ref: https://stats.stackexchange.com/questions/95209/how-can-i-interpret-sklearn-confusion-matrix
    print('tpr: ', tpr, ', fpr: ', fpr)
    pd.crosstab(y_true, resultFineClus, rownames=['True'], colnames=['Predicted'], margins=True)
    
    #auc = roc_auc_score(y_true, scoreOneTable)
    #aucClus = auc
    
    ##for checking logic error
    #print('Auc: ', auc)
    #threshold = oneThreshold(thresholdArr,strategy)
    #print('thresholdClus: ',threshold)
    #suspicious_plot(scoreOneTable, y_true, threshold)
    
    return fineClus, tprClus, fprClus, thresholdArr, scoreOneTable, fineClus, feaNumArr, strategy, betaR_Clus, uArr, dArr, A_m_Arr

## mean +- d*std
def test_paper(X_train, X_vali, y_vali,
               feaNumArr, X_test, y_test,
               sRateW, ini_For_W, dropRateT_W,
               sRateClus, ini_For_Clus, dropRateT_Clus,
               k, thresLowBDD, strategy,
               NTRAIL=5,
               thresUpBDD=1,
               RANDOMPICK=False,
               UPDATE=True):
    betaW = 1/(sRateW*len(X_train))
    
    ## train again to get optimal model
    ## through best parameters combination
    tpr, fpr, threshold, scoreTable, betaR_W, u, d, A_m = trainValiWhole_paper(X_train, X_vali, y_vali,
                                                                       sRateW, betaW, ini_For_W, dropRateT_W,
                                                                       UPDATE=UPDATE)
    
    
    ## using mean+2std to determine thershold
    print('Determine threshold')
    
    #scoreMean = np.mean(scoreTable)
    #scoreStd = np.std(scoreTable)
    
    #thresholdW = scoreMean + 0*scoreStd    
    print('Directly using remain top')
    thresholdW = threshold
    print('Threshold: ', thresholdW)
    
    testA = X_test.astype(float).values.copy()
    ## testing
    result, scoreTable = test_pca(testA, u, d, A_m, sRateW, betaR_W, thresholdW, UPDATE = True, upFor = 1)
    
    tn, fp, fn, tp = confusion_matrix(y_test, result).ravel()
    
    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)
    
    print('Original pca: ')
    print('Tpr: ', tpr, ', fpr: ', fpr)
    
    ## ------------------------------------------------------------------------------------------------
    # Laplacian
    
    betaClus = 1/(sRateClus*len(X_train))
    
    fineClus, tprClus, fprClus, thresholdArr, scoreOneTable, fineClus, feaNumArr, strategy, betaR_Clus, uArr, dArr, A_m_Arr = trainValiClus(X_train, X_vali, y_vali,
                       sRateClus, betaClus, ini_For_Clus, dropRateT_Clus,
                       k, thresLowBDD, strategy,
                       thresUpBDD=1,
                       UPDATE=True)
    
    ## determine threshold for Laplacian
    #scoreMeanCl = np.mean(scoreOneTable)
    #scoreStdCl = np.std(scoreOneTable)
    #thresholdCluster = scoreMeanCl + 0*scoreStdCl
    
    #for clus in range(0,k):
    #    thresholdArr[clus] = thresholdCluster
    print('For Laplacian, directly using remaining top')
    print(thresholdArr)
    
    resultPack = test_pca_mul_select(fineClus, feaNumArr, k, strategy, testA, uArr, dArr, A_m_Arr, 
                                     sRateClus, betaR_Clus, thresholdArr,
                                    UPDATE=UPDATE)
    resultFineClus = resultPack[0]
    scoringTable = resultPack[1]
    
    tn, fp, fn, tp = confusion_matrix(y_test, resultFineClus).ravel()
    tprClus = tp/(tp+fn)
    fprClus = fp/(tn+fp)
    
    print('Laplacian, In detection: ')
    print('Tpr: ', tprClus, ', fpr: ', fprClus)
    
    
    
    #return aucWhole, tprWhole, fprWhole, aucLap, tprLap, fprLap, thresDescend
    return tpr, fpr, tprClus, fprClus
###



def procedure_paper():
    print('Training')
    
    print('Validation')
    
    print('get best parameter, determine threshold')
    
    print('Testing')
    
    



###    
    
    
    































