#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster,cophenet

def cophenetic(X_data) :
    """
    Calculate cophenetic correlation coefficient of linkages

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)

    Returns
    -------
    None
    """
    abs_corre = abs(X_data.corr())
    Z1 = linkage(abs_corre, method='average')
    Z2 = linkage(abs_corre, method='complete')
    Z3 = linkage(abs_corre, method='single')
    c1, coph_dists1 = cophenet(Z1, pdist(abs_corre))
    c2, coph_dists2 = cophenet(Z2, pdist(abs_corre))
    c3, coph_dists3 = cophenet(Z3, pdist(abs_corre))
    print('average linkage cophenet:', c1)
    print('complete linkage cophenet:',c2)
    print('single linkage cophenet:',c3)


class FeatureCluster:
    """
    Make features(decriptors) clusters based on hierarchical clustering method

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    link : str , kind of linkage (single or complete or average)
    cut_d : int, depth in cluster(dendrogram)

    Sub functions
    -------
    set_cluster(self)
    cluster_dist(self)
    """
    def __init__(self,X_data,link,cut_d):
        self.cluster_info = []
        self.assignments = 0
        self.X_data = X_data
        self.abs_corre = abs(X_data.corr())
        self.link = link
        self.cut_d = cut_d

    def set_cluster(self) :
        """
        Make input of feature selection function

        Returns
        -------
        assignments : dic, shape = (n_features)
        return cluster information as a input of feature selection function
        """

        Z = linkage(self.abs_corre, method=self.link)
        self.assignments = fcluster(Z,self.cut_d,'distance')
        cluster_output = DataFrame({'Feature':list(self.X_data.columns.values) , 'cluster':self.assignments})
        nc = list(cluster_output.cluster.values)
        nnc = max(nc)
        name = list(cluster_output.Feature.values)
        cludic = {}
        for i in range(0,len(nc)):
            k = name[i]
            v = nc[i]
            cludic[k] = v
        for t in range(1,nnc+1):
            vv = []
            vv = [key for key, value in cludic.items() if value == t]    #Find Key by Value in Dictionary
            self.cluster_info.append(vv)
            print('\n','\x1b[1;46m'+'Cluster'+'\x1b[0m',t,vv,)
        return self.assignments

    # cluster correlation coefficient distribution
    def cluster_dist(self) :
        """
        Show dendrogram of correlation coefficient distribution of each cluster

        Returns
        -------
        None
        """
        dist_box = []

        assignments = self.cluster_info
        cluster_output = DataFrame({'Feature':list(self.X_data.columns.values) , 'cluster':self.assignments})
        nc = list(cluster_output.cluster.values)
        name = list(cluster_output.Feature.values)

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

        for s in cluster :
            desc_set = s
            c = len(desc_set)
            if c == 1 :
                pass
            else:
                tay = self.abs_corre.loc[desc_set,desc_set]
                t =  np.array(tay)

                av = (t.sum()-c)/2
                aver = av/((c*c-c)/2)
                dist_box.append(aver)
        plt.hist(dist_box)
        plt.ylabel('Frequency')
        plt.xlabel('Correlation coefficient of each cluster')
