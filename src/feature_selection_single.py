#-*- coding: utf-8 -*-
import datetime
import random
import copy
import math
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from numpy import ndarray
import sklearn.linear_model as lm
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.svm import SVC


def selection(X_data,y_data,cluster_info,model='regression',learning=500000,bank=200,component=4,pntinterval=1000):
    """
    Feature selection algorothm with single core

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    y_data : pandas DataFrame , shape = (n_samples,)
    cluster_info : Return value of clustering.FeatureCluster.set_cluster()
    model : default='regression' / when user use default value, operate this function for regression model /
    when user use other value like 'Classification', operate this function for classification model/
    This function only have two options : regression(default) or classification
    learning : Number of learning
    bank : Size of bank(pool) in selection operation
    component : Number of feature in prediction model
    pntinterval : print currnt score and selected features set once every 'pntinterval' times

    Returns
    -------
    list, result of selected best feature set
    """
    now = datetime.datetime.now()
    nowTime = now.strftime('%H:%M:%S')
    print('Start time : ',nowTime)

    if model == 'regression' :
        print('\x1b[1;42m'+'Regression'+'\x1b[0m')
        y_mlr = lm.LinearRegression()
        e_mlr = lm.LinearRegression()
    else :
        print('\x1b[1;42m'+'Classification'+'\x1b[0m')
        y_mlr = SVC(kernel='rbf', C=1.0 ,gamma=0.1 ,random_state=0)
        e_mlr = SVC(kernel='rbf', C=1.0 ,gamma=0.1 ,random_state=0)


    assignments = cluster_info
    cluster_output = DataFrame({'Features':list(X_data.columns.values) , 'cluster':assignments})
    nc = list(cluster_output.cluster.values)
    name = list(cluster_output.Features.values)

    clu_hist  = {}
    cluster= []

    for i in range(0,len(nc)):
        k = name[i]
        v = nc[i]
        clu_hist [k] = v
    for t in range(1,max(nc)+1):
        vv = []
        vv = [key for key, value in clu_hist.items() if value == t]    #Find Key by Value in Dictionary
        cluster.append(vv)

    leen = list(range(max(nc)))

    ini_desc_bank= []
    index_sort_bank = []

    while len(ini_desc_bank) < bank :
        leenlis = list(leen)
        ini_desc = []
        index_sort = []

        for j in range(component) :
            clu_n = random.choice(leenlis)
            leenlis.remove(clu_n)

            index_sort.append(clu_n)
            ini_desc.append(random.choice(cluster[clu_n]))
        index_sort.sort()
        ini_desc.sort()

        if index_sort not in index_sort_bank :
            index_sort_bank.append(index_sort)
            ini_desc_bank.append(ini_desc)

    uni_desc = []
    for i in ini_desc_bank :
        x = DataFrame(X_data, columns=i)
        xx = np.array(x, dtype=np.float64)
        y = np.array(y_data, dtype=np.float64)

        #y_mlr = lm.LinearRegression()
        y_mlr.fit(xx,y_data.values.ravel())
        score = y_mlr.score(xx,y)
        mid = []
        mid.append(score)
        mid.append(i)

        uni_desc.append(mid)

    top_ranker = uni_desc
    n = 0
    while n < learning :
        n = n + 1
        evoldes = []
        for h in top_ranker :
            evoldes.append(h)


        for j,i in top_ranker :    # j = r2 , i = desc set
            #child = []
            group_n = []
            for y in i :
                gn= clu_hist[y]-1
                group_n.append(gn)

            sw_index = random.randrange(0,component)
            while 1 :
                sw_group = random.randrange(0,max(nc))
                if sw_group in group_n :
                    continue

                else :
                    break

            b_set = copy.deepcopy(i)
            b_set[sw_index] = random.choice(cluster[sw_group])
            b_set.sort()
            x = DataFrame(X_data, columns=b_set)
            xx = np.array(x, dtype=np.float64)
            y = np.array(y_data, dtype=np.float64)
            #e_mlr = lm.LinearRegression()
            e_mlr.fit(xx,y_data.values.ravel())
            score = e_mlr.score(xx,y)
            mid = []
            mid.append(score)
            mid.append(b_set)
            evoldes.append(mid)
        #print evoldes ,len(evoldes)

        rank_filter = []
        #중복지우는부분
        for i in evoldes :
            if i in rank_filter :
                pass
            else :
                rank_filter.append(i)

        rank_filter.sort(reverse= True)
        top_ranker =  rank_filter[:bank]

        if n % pntinterval == 0 :
            tt = datetime.datetime.now()
            print(n , '=>', tt.strftime('%H:%M:%S') , top_ranker[0])

    print(top_ranker[0])
    clulog = []
    for j,i in top_ranker :
            #child = []
            group_n = []
            for y in i :
                gn= clu_hist[y]
                clulog.append(gn)

            break
    print("Model's cluster info",clulog)
    fi = datetime.datetime.now()
    fiTime = fi.strftime('%H:%M:%S')
    print('Finish time : ',fiTime)
    return top_ranker[0][1]
