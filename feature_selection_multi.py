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
from joblib import Parallel, delayed


def selector(X_data,y_data,cluster_info,model='regression',learning=5000,bank=200,component=4):
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

        for j in range(component):
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

    top_ranker = uni_desc  #R2
    n = 0
    while n < learning :
        n = n + 1
        evoldes = []
        for h in top_ranker :
            evoldes.append(h)


        for j,i in top_ranker :    # j = r2 , i = desc set
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
            e_mlr.fit(xx,y_data.values.ravel())
            score = e_mlr.score(xx,y)
            mid = []
            mid.append(score)
            mid.append(b_set)
            evoldes.append(mid)


        rank_filter = []

        for i in evoldes :
            if i in rank_filter :
                pass
            else :
                rank_filter.append(i)


        rank_filter.sort(reverse= True)
        top_ranker =  rank_filter[:bank]

    return top_ranker[0]

class MultiSelection :
    """
    Features selection algorothm with mult processing

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
    component : Number of features in prediction model

    Sub functions
    -------
    run (self, n_core=3, run_job=3)
    """

    def __init__(self, X_data,y_data,cluster_info,model='regression',learning=5,bank=100,component=3) :
        self.X_data = X_data
        self.y_data = y_data
        self.cluster_info= cluster_info
        self.learning = learning
        self.bank = bank
        self.component =component
        self.model = model
        if self.model == 'regression' :
            print('\x1b[1;42m'+'Regression'+'\x1b[0m')
        else :
            print('\x1b[1;42m'+'Classification'+'\x1b[0m')

    def run (self, n_core=3, run_job=3) :
        """
        Parameters
        ----------
        n_core : Number of used cores
        run_job : Number of times to perform the selection function

        Returns
        -------
        list, result of selected best Feature set
        """
        merger = Parallel(n_jobs=n_core)(delayed(selector)
                                        (self.X_data,self.y_data,self.cluster_info,model=self.model,learning=self.learning,
                                        bank=self.bank,component=self.component) for i in range(run_job))

        merger.sort(reverse= True)
        for m in merger:
            print(m)
        return merger[0][1]
