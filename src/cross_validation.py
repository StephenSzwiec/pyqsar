#-*- coding: utf-8 -*-
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
# need NOT scaled data
def k_fold(X_data,y_data,feature_set,run=100,k=5) :
    """
    Repeat K-fold 'run' times and summary the best implementation.

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    y_data : pandas DataFrame , shape = (n_samples,)
    feature_set : list, set of features that make up model
    run : int, number of implementation
    k : int, k of 'K'-Fold

    Returns
    -------
    None
    """
    gingerbreadman = []
    n=0
    while n < run :
        n = n+1
        Q2=[]
        R2 =[]
        coef = []
        intercept = []
        trainset_index=[]
        testset_index=[]
        x = X_data.loc[:,feature_set].as_matrix()
        y =y_data.as_matrix()
        kf = KFold(n_splits=k,shuffle=True)
        predY = np.zeros_like(y)
        for train,test in kf.split(x):
            scaler = MinMaxScaler()
            scaler.fit(x[train])
            xtrain = scaler.transform(x[train])
            xtest = scaler.transform(x[test])
            clf = LinearRegression()
            clf.fit(xtrain,y[train])
            predY[test] = clf.predict(xtest)
            rs = clf.score(xtrain,y[train])
            qs = clf.score(xtest,y[test])
            coe = clf.coef_
            inte = clf.intercept_
            Q2.append(qs)
            R2.append(rs)
            coef.append(coe)
            intercept.append(inte)
            trainset_index.append(train)
            testset_index.append(test)
        rmse = np.sqrt(mean_squared_error(predY,y))
        maxq2 = np.max(np.array(Q2))
        index = Q2.index(maxq2)
        mid = []
        mid.append(np.mean(np.array(Q2))) #ginger[0]
        mid.append(np.mean(np.array(R2))) #ginger[1]
        mid.append(rmse) #ginger[2]
        mid.append(coef[index]) #ginger[3]
        mid.append(intercept[index]) #ginger[4]
        mid.append(trainset_index[index])
        mid.append(testset_index[index])
        gingerbreadman.append(mid)

    gingerbreadman.sort()
    best =  gingerbreadman[-1]



    print('R^2CV mean: {:.6}'.format(best[1]))
    print('Q^2CV mean: {:.6}'.format(best[0]))
    print('RMSE CV : {:.6}'.format(best[2]))
    print('Features set =', feature_set)
    print('Model coeff = ',best[3])
    print('Model intercept = ',best[4])

    train_ind = best[5]
    test_ind = best[6]
    #print trainind,testind

    pred_plotY = np.zeros_like(y)
    g_mlr = LinearRegression()
    g_mlr.fit(x[train_ind],y[train_ind])
    pred_plotY[train_ind] = g_mlr.predict(x[train_ind])
    pred_plotY[test_ind] = g_mlr.predict(x[test_ind])
    plt.ylabel("Predicted Y")
    plt.xlabel("Actual Y")
    plt.scatter(y[train_ind],pred_plotY[train_ind],color=['gray'])
    plt.scatter(y[test_ind],pred_plotY[test_ind],color=['red'])
    plt.plot([y.min() , y.max()] , [[y.min()],[y.max()]],"black" )
    plt.show()
