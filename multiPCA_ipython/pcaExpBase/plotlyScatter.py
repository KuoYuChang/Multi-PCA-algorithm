from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import decomposition

import numpy as np
import pandas as pd


plt.rcParams["figure.figsize"][0] = 9
plt.rcParams["figure.figsize"][1] = 6


def get_plotItems(pStr='tryout'):

    colors = ['rgb(55,126,184)','rgb(228,26,28)','rgb(77,175,74)']

    layout = dict(
        width=800,
        height=550,
        autosize=False,
        title=pStr,
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectratio = dict( x=1, y=1, z=0.7 ),
            #aspectmode = 'manual'        
        ),
    )
    
    return colors, layout


## define function
## plot both data and label
## using plotly for interactive

# x should be dataframe, y should be series or np array
# X: original, Xpr: projected

# no need title
def plotly_PCAproj_xy(X, y_label, 
                      colors, layout, 
                      n_comp = 3):

    Xtemp = X.values.copy()
    
    pca = decomposition.PCA(n_components=n_comp)
    pca.fit(Xtemp)
    # maybe can put separately
    Xpr = pca.transform(Xtemp)

    # for safety, using numpy tool after transform
    #y_temp = y.values.copy()
    
    #Xpr_n = Xpr[np.where(y_temp == 0)[0]]
    #Xpr_a = Xpr[np.where(y_temp == 1)[0]]
    
    ## running
    data = []
    pointName = ['normal','anomaly']

    for i in range(0, 2):
        # 0 normal 1 anomaly
        temp = X.loc[y_label == i]
        temp_now = pca.transform(temp)

        color = colors[i]
        x = temp_now[:,0]
        y = temp_now[:,1]
        z = temp_now[:,2]

        trace = dict(
                name = pointName[i],
                x = x, y = y, z = z,
                type = "scatter3d",    
                mode = 'markers',
                marker = dict( size=2, color=color, line=dict(width=0) ) )
        data.append(trace)
    len(data)    
    # plot
    fig = dict(data=data, layout=layout)

    # IPython notebook
    iplot(fig, filename='implement', validate=False)
    
    
# 3-D plot
# X as dataFrame
# maybe no title first
def plotly_3D_xy(X, y_label, 
                 colors, layout):

    data = []
    pointName = ['normal','anomaly']

    for i in range(0, 2):
        # 0 normal 1 anomaly
        temp_now = X.loc[y_label == i].copy()
        print(i, len(temp_now), len(X))

        name = pointName[i]
        color = colors[i]
        x = temp_now.iloc[:,0].values.copy()
        y = temp_now.iloc[:,1].values.copy()
        z = temp_now.iloc[:,2].values.copy()

        trace = dict(
                name = name,
                x = x, y = y, z = z,
                type = "scatter3d",    
                mode = 'markers',
                marker = dict( size=2, color=color, line=dict(width=0) ) )
        
        data.append(trace)  
    # plot
    fig = dict(data=data, layout=layout)

    # IPython notebook
    iplot(fig, filename='implement', validate=False)

    
## 2-D plot
# X as dataFrame
## for 2D, original plt is enough
def plot2D_xy(X, y):
    # normal
    Xpr_n = X.loc[y == 0].values.copy()
    # anomaly
    Xpr_a = X.loc[y == 1].values.copy()
    
    ax = plt.axes()

    
    plt.plot(Xpr_n[:,0], Xpr_n[:,1], 'co')
    plt.plot(Xpr_a[:,0], Xpr_a[:,1], 'ro')
    
    
    #plt.plot(Xpr[:,0], Xpr[:,1])
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])

    plt.show()

    
def plot_cluster_k(clusKey, feaNumArr, currentData, tempY, colors, layout):
    #clusKey = 4
    #X = X_train.iloc[:, featureNum]
    featureNum = feaNumArr[clusKey]

    X = currentData.iloc[:, featureNum]
    #tempY = y_label[idxT]

    #print("Cluster ", clusKey)
    print(featureNum)
    if len(featureNum) > 3:
        print("PCA into 3 dim")
        plotly_PCAproj_xy(X, tempY, colors, layout)
    elif len(featureNum) == 3:
        print("3D plot")
        plotly_3D_xy(X, tempY, colors, layout)
    elif len(featureNum) == 2:
        plot2D_xy(X, tempY)
    else:
        print("In ", clusKey, ", only 1 dimension")
        
        
def randomPick(dataX, y_label, anofrac, norfrac=1, PICKNORMAL=False):
    
    X_index = dataX.index
    y_Series = pd.Series(y_label, index=X_index)
    y_Series = y_Series.rename('label')
    y_Series.head()

    dataPlotTemp = pd.concat([dataX,y_Series],axis=1)

    ## pick all normal
    dataPlotNorT = dataPlotTemp.loc[dataPlotTemp['label']==0]
    dataPlotAnoT = dataPlotTemp.loc[dataPlotTemp['label']==1]

    if PICKNORMAL:
        ## also pick some noraml, prevent out of memory
        print('also random pick normal')
        dataPlotNorT2 = dataPlotNorT.sample(frac=norfrac)
        dataPlotAno = dataPlotAnoT.sample(n=int(len(dataPlotNorT2)*anofrac))
        dataPlot = pd.concat([dataPlotNorT2,dataPlotAno],axis=0)
    else:
        dataPlotAno = dataPlotAnoT.sample(n=int(len(dataPlotNorT)*anofrac))
        dataPlot = pd.concat([dataPlotNorT,dataPlotAno],axis=0)

    

    y_plot = dataPlot['label']
    dataPlotX = dataPlot.drop(['label'],axis=1)
    
    return dataPlotX, y_plot