# for showLaplacian
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.sparse import csgraph
from sklearn.cluster import spectral_clustering

import matplotlib.patches as patches

from .whuir_spectral import spec_clus_Lrw
from .whuir_spectral import spec_get_L
#from osPcaLaplacian import scaleTo01
#from osPcaLaplacian import scaleToN01



def getNormal(currentData, y_label):
    X_index = currentData.index
    y_Series = pd.Series(y_label, index=X_index)
    y_Series = y_Series.rename('label')
    
    X_whole = pd.concat([currentData,y_Series],axis=1)
    
    X_nor_temp = X_whole.loc[X_whole['label']==0]
    X_nor = X_nor_temp.drop(['label'],axis=1)
    return X_nor

def showLaplacian(currentData, Name, NOR_RAN=False, SCALE=0, rate=0.5, pickN=100):
    plt.rcParams["figure.figsize"][0] = 9
    plt.rcParams["figure.figsize"][1] = 6
    
    tempData = currentData.sample(frac=rate)
    
    ## check if features that std==0
    ## no more dealing since only for plot
    
    # take away features that std==0 first
    STD_DES = tempData.describe().loc['std']
    zeros = np.where(STD_DES == 0)[0]

    ## may need to reorder columns
    if len(zeros)>0:
        print('Drop some features that std==0')
        print(zeros)
        std0_part = tempData.iloc[:,zeros]
        restData = tempData.drop(zeros,axis=1)
    else:
        restData = tempData
    
    #STD_DES = restData.describe().loc['std']
    #zeros = np.where(STD_DES == 0)[0]
    #print('restData: ', zeros)
    
    #scale data if need
    if SCALE==1:
        print('scale to 01')
        tempData = (tempData-tempData.min()) / (tempData.max()-tempData.min())
    elif SCALE==2:
        print('scale to N01')
        tempData = (tempData-tempData.mean()) / (tempData.std())
    
    
    tempW = restData.corr().values
    weight_abs = abs(tempW)
    #w_abs
    plt.imshow(weight_abs)
    #show scale of colors
    plt.colorbar()
    plt.show()
    
    
    # save abs of weight matrix
    saveWeight = pd.DataFrame(weight_abs)
    saveWeight.to_csv('plotInfo/'+Name+'_weightAbs.csv', index=False, header=None)
    
    #############################
    
    #from scipy.sparse import csgraph

    ## note: from github, normalized Laplacian is symmetric
    ## choose normalized random walk or noramlized symmetric
    if NOR_RAN:
        Lsym = spec_get_L(weight_abs, NOR_RAN=NOR_RAN)
    else:
        Lsym = csgraph.laplacian(weight_abs, normed = True)

    # eigen decomposition
    from numpy import linalg as LA
    w, v = LA.eig(Lsym)
    sortW = np.sort(w) # no need order here

    ## plot out
    ## adjust plot parameter

    tempX = np.arange(len(sortW))+1

    plt.scatter(tempX[1:15], sortW[1:15])
    #plt.plot(tempX, sortW)
    plt.plot(tempX[1:15], sortW[1:15])
    
    plt.xlabel('Index of Eigenvalues', fontsize=20)
    plt.ylabel('Eigenvalues', fontsize=20)
    
    plt.show()
    
    # save eignevalues
    saveSortW = pd.DataFrame(sortW)
    saveSortW.to_csv('plotInfo/'+Name+'_eigenvalues.csv', index=False, header=None)
    #####################################
    
    
    ## plot eigen difference

    diffSortW = np.diff(sortW)

    tempX = np.arange(len(diffSortW))+1

    # bar plot
    plt.bar(tempX, diffSortW)
    plt.show()
    
    ## show top candidates of k choices
    ## see eigen gap, find from big to small
    ## just print out at first
    ## for i in diffSortW, means eigen i+1 - i
    ## but python start from 0, so add 1 
    rank0 = np.argsort(-diffSortW)
    rank = rank0[0:pickN]+1
    
    #print(rank0[0:pickN])
    print(rank)
    
    
    

    return weight_abs, diffSortW, rank

def quickShowLaplacian(Name, NOR_RAN=False, rate=0.5, pickN=100):
    tempW = pd.read_csv('plotInfo/'+Name+'_weightAbs.csv', header=None)
    weight_abs = tempW.values.copy()
    plt.imshow(weight_abs)
    #show scale of colors
    plt.colorbar()
    plt.show()
    
    sortW_CSV = pd.read_csv('plotInfo/'+Name+'_eigenvalues.csv', header=None)
    sortW = sortW_CSV.values.copy()
    tempX = np.arange(len(sortW))+1
    plt.scatter(tempX[1:15], sortW[1:15])
    #plt.plot(tempX, sortW)
    plt.plot(tempX[1:15], sortW[1:15])
    
    plt.xlabel('Index of Eigenvalues', fontsize=20)
    plt.ylabel('Eigenvalues', fontsize=20)
    
    plt.show()
    

def colReorder(data, REORDER=False, colOrder=np.zeros(1)):
    if REORDER:
        print('Reorder columns of dataset')
        data = data.iloc[:,colOrder]
        
    
    return data


# might have no choice to choose k = 2
## maybe modified, input dataFrame
def spectral_result(X_nor, k_clusters, Name, NOR_RAN=False):
    ## dealing for every a training data
    ## be responsible for some heavy works
    
    ### check if any a feature that std==0
    STD_DES = X_nor.describe().loc['std']
    zeros = np.where(STD_DES == 0)[0]
    
    tempCol = np.arange(len(X_nor.columns))
    
    
    #print(zeros)
    if len(zeros)>0:
        print('number: ', len(zeros))
        print('Some features std==0')
        print(zeros)
        std0_part = X_nor.iloc[:,zeros]
        X_nor = X_nor.drop(zeros,axis=1)
        
        ## storage column order
        backPart = tempCol[zeros]
        frontPart = np.delete(tempCol, zeros)
        colOrder = np.concatenate((frontPart,backPart))
        #print(backPart, frontPart, colOrder)
        REORDER = True
    else:
        colOrder = tempCol
        REORDER = False
    #end if
    
    weight_abs = abs(X_nor.corr().values)
    
    
    ## make sure that correlation matrix contains no nan
    ## notice that may choose normalized random walk, control with boolean variable?
    if NOR_RAN:
        labels = spec_clus_Lrw(weight_abs, k_clus=k_clusters, NOR_RAN=NOR_RAN)
    else:
        labels = spectral_clustering(weight_abs, n_clusters=k_clusters, eigen_solver='arpack')
    

    ## finally original dataset need to be rerodered columns(if needed)
    #currentData = pd.concat([restData,std0_part],axis=1)
    
    ## get cluster index once and for all
    feaNumArr = [None]*k_clusters
    feaNumSize = np.zeros(k_clusters)
    for clusKey in range(0, k_clusters):
        featureNum = np.where(labels == clusKey)[0]
        feaNumArr[clusKey] = featureNum
        
        feaNumSize[clusKey] = len(featureNum)
    ## turn off if ok
    print(feaNumArr)
    
    ## fill in largest cluster if needed
    if REORDER:
        print('Put into the largest cluster')
        LarKey = np.argmax(feaNumSize)
        print('The largest key: ', LarKey)
        std0_P = np.arange(len(zeros))+len(labels)
        #print(len(zeros), len(labels), std0_P)
        enLarge = np.concatenate((feaNumArr[LarKey],std0_P))
        feaNumArr[LarKey] = enLarge
    
    # save feaNumArr
    saveFeaNum = pd.DataFrame(feaNumArr)
    saveFeaNum.to_csv('plotInfo/'+Name+'_feaNumArr.csv', index=False, header=None)
    ####################################################
    
    return feaNumArr, labels, REORDER, colOrder


def readFeaNumArr(Name):
    tempArr = pd.read_csv('plotInfo/'+Name+'_feaNumArr.csv', header=None)
    
    #print(tempArr)
    
    feaNumArr = []
    k = len(tempArr)
    
    for i in range(0,k):
        tempRow = tempArr.iloc[i]
        
        feaNum = tempRow[~pd.isnull(tempRow)]
        feaNumber = feaNum.values.copy()
        feaNumArr.append(feaNumber)
    #end for
    
    return feaNumArr
    


def reArrangeW(feaNumArr, currentDataTemp, k, SCALE=0, REORDER=False, colOrder=np.zeros(1)):
    ## show correlation matrix with rearrangement
    ## modified: direct input feaNumArr
    
    if REORDER:
        currentData = colReorder(currentDataTemp, REORDER=REORDER, colOrder=colOrder)
    else:
        currentData = currentDataTemp
        
    #scale here
    #scale data if need
    if SCALE==1:
        print('scale to 01')
        currentData = (currentData-currentData.min()) / (currentData.max()-currentData.min())
    elif SCALE==2:
        print('scale to N01')
        currentData = (currentData-currentData.mean()) / (currentData.std())

    arrange = np.zeros(len(currentData.columns))
    idxH = 0
    # rearrenge
    for clusKey in range(0, k):
        #featureNum = np.where(labels == clusKey)[0]
        featureNum = feaNumArr[clusKey]

        
        idxE = idxH + len(featureNum)

        # become float type
        arrange[idxH:idxE] = featureNum

        #updata index
        idxH = idxE

        #print(arrange)

    # transform back to integer
    arrange = arrange.astype(int)
    
    # indexing with iloc
    reArrData = currentData.iloc[:, arrange]
    tempW = reArrData.corr().values
    weight_abs = abs(tempW)
    
    
    # Create figure and axes
    fig,ax = plt.subplots(1)
    
    
    #w_abs
    ax.imshow(weight_abs)
    #show scale of colors
    #plt.colorbar()
    
    
    ## add blocks here
    # Create a Rectangle patch
    ## adjust 0.5, by Yang (ˊ● ω ●ˋ)
    sPoint = np.zeros(2)-0.5
    for clusKey in range(0, k):
        #featureNum = np.where(labels == clusKey)[0]
        featureNum = feaNumArr[clusKey]
        width = len(featureNum)
        
        rect = patches.Rectangle(sPoint, width, width,linewidth=1,edgecolor='white',facecolor='none')
        ax.add_patch(rect)

        
        sPoint = sPoint + width
    plt.show()