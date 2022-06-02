#-*- coding: utf-8 -*-
# for classification model_selection

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
from sklearn import datasets
from sklearn.metrics import mean_squared_error , r2_score, accuracy_score
from sklearn.svm import SVC

class ClassifierScore :
    """
    Provides summary information about Classification

    Parameters
    ----------
    y_data : pandas DataFrame , shape = (n_samples,)
    pred_y : pandas DataFrame , shape = (n_samples,)
    => predicted Y values as result of classification

    Sub functions
    -------
    score (self)
    tf_table(self)
    """

    def __init__ (self,y_data,pred_y) :
        self.y_data = y_data
        self.pred_y = pred_y
        self.real_y = [] #hash y_data

        for i in np.array(self.y_data) :
            self.real_y.append(i[0])

    def score (self) :
        """
        Calculate accuracy score

        Returns
        -------
        None
        """
        n = 0
        cnt = 0
        for i in np.array(self.real_y) :
            if i != self.pred_y[n] :
                cnt += 1
            n += 1
        print('Number of all :',n) #all data
        print('Number of worng :', cnt)
        print('AccuracyScore :',accuracy_score(self.real_y, self.pred_y))

    def tf_table(self) :
        """
        Calculate Precision & Recall

        Returns
        -------
        None
        """
        one = 0
        zero = 0
        n = 0
        cnt = 0
        realzero = 0
        realone = 0
        for i in np.array(self.y_data) :
            if i[0] == 0 :
                zero += 1
            if i[0] == 1 :
                one += 1

        for i in np.array(self.y_data):
            if i[0] != self.pred_y[n]:
                #print ('real',i[0],'///','pred',y_pred[n])
                if i[0] == 0 :
                    realzero += 1
                if i[0] == 1 :
                    realone += 1
                cnt +=1
            n += 1


        print(('Number of 1 :',one))
        print('Number of 0 :',zero)
        print('True Positive(real 1 but pred 1) :',one-realone) #TP
        print('True Negative(real 0 but pred 0) :',zero-realzero) #TN
        print('False Positive(real 0 but pred 1) :',realzero) #FP
        print('False Negative(real 1 but pred 0) :',realone)  #FN
        print('Precision', (one-realone)/((one-realone)+realzero)) # TP / TP+FP
        print('Recall',(one-realone)/((one-realone)+realone)) #  TP / TP+FN
