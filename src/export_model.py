#-*- coding: utf-8 -*-

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt

from bokeh.plotting import figure, output_file, show, ColumnDataSource, output_notebook
from bokeh.models import HoverTool, BoxSelectTool


class ModelExport :
    """
    Summary model information

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    y_data : pandas DataFrame , shape = (n_samples,)
    feature_set : list, set of features that make up model

    Sub functions
    -------
    train_plot(self)
    train_plot_inter(self)
    mlr(self)
    features_table(self)
    model_corr(self)
    """
    def __init__ (self, X_data, y_data, feature_set) :
        self.X_data = X_data
        self.y_data = y_data
        self.feature_set = feature_set

    def train_plot(self) :
        """
        Show prediction training plot

        Returns
        -------
        None
        """
        x = self.X_data.loc[:,self.feature_set].as_matrix()
        y = self.y_data.as_matrix()
        pred_plotY = np.zeros_like(y)
        g_mlrr = LinearRegression()
        g_mlrr.fit(x,y)
        pred_plotY = g_mlrr.predict(x)
        plt.ylabel("Predicted Y")
        plt.xlabel("Actual Y")
        plt.scatter(y,pred_plotY,color=['gray'])
        plt.plot([y.min() , y.max()] , [[y.min()],[y.max()]],"black" )
        plt.show()

    def train_plot_inter(self) :
        """
        Show prediction training interactive plot

        Returns
        -------
        None
        """
        # index start from 0
        output_notebook()
        TOOLS = [BoxSelectTool(), HoverTool()]
        x = self.X_data.loc[:,self.feature_set].as_matrix()
        Ay = self.y_data.as_matrix()
        ipred_plotY = np.zeros_like(Ay)
        ig_mlrr = LinearRegression()
        ig_mlrr.fit(x,Ay)
        Py = ig_mlrr.predict(x)
        ppy = []
        aay = []
        for i in Py :
            ppy.append(i[0])
        for j in Ay :
            aay.append(j[0])
        p = figure(plot_width=600, plot_height=600, tools=TOOLS, title="Predicted & Actual")
        p.yaxis.axis_label = "Predicted Y"
        p.xaxis.axis_label = "Actual Y"
        p.circle(aay,ppy,size=20, color="orange", alpha=0.5 )
        show(p)

    def mlr(self) :
        """
        c model information with result of multiple linear regression

        Returns
        -------
        None
        """
        x = self.X_data.loc[:,self.feature_set].as_matrix()
        y = self.y_data.as_matrix()
        mlr = LinearRegression()
        mlr.fit(x,y)
        print('Model features: ',self.feature_set)
        print('Coefficients: ', mlr.coef_)
        print('Intercept: ',mlr.intercept_)

        #MSE
        #print "MSE: %.3f" % np.mean((mlr.predict(x) - y) ** 2)
        #print mean_squared_error(mlr.predict(x),y)
        print("RMSE: %.6f" % np.sqrt(mean_squared_error(mlr.predict(x),y)))
        # Explained variance score
        print('R^2: %.6f' % mlr.score(x, y))

    def features_table(self) :
        """
        Show feature vlaues table

        Returns
        -------
        table
        """
        desc = DataFrame(self.X_data, columns=self.feature_set)
        result = pd.concat([desc, self.y_data], axis=1, join='inner')
        return result

    def model_corr(self) :
        """
        Correlation coefficient of features table

        Returns
        -------
        table
        """
        X = DataFrame(self.X_data, columns=self.feature_set)
        result = pd.concat([X, self.y_data], axis=1, join='inner')
        pd.plotting.scatter_matrix (result, alpha=0.5, diagonal='kde')
        return result.corr()

def external_set(X_data,y_data,exdataX,exdataY,feature_set) :
    """
    Presiction external data set

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    y_data : pandas DataFrame , shape = (n_samples,)
    exdataX :pandas DataFrame , shape = (n_samples, n_features)
    => External data set x
    exdataY :pandas DataFrame , shape = (n_samples,)
    => External data set y
    feature_set : list, set of features that make up model

    Returns
    -------
    None
    """
    x = X_data.loc[:,feature_set].as_matrix()
    y = y_data.as_matrix()
    exd = exdataX.loc[:,feature_set].as_matrix()
    exdY = exdataY.as_matrix()

    scalerr = MinMaxScaler()
    scalerr.fit(x)

    x_s = scalerr.transform(x)
    exd_s = scalerr.transform(exd)
    mlrm = LinearRegression()
    mlrm.fit(x_s,y)
    trainy= mlrm.predict(x_s)
    expred = mlrm.predict(exd_s)
    print('Predicted external Y \n',expred)
    print('R2',mlrm.score(x_s,y))
    print('external Q2',mlrm.score(exd_s,exdY))
    print('coef',mlrm.coef_)
    print('intercept',mlrm.intercept_)
    plt.ylabel("Predicted Y")
    plt.xlabel("Actual Y")
    plt.scatter(y,trainy,color=['gray'])
    plt.scatter(exdY,expred,color=['red'])
    plt.plot([y.min() , y.max()] , [[y.min()],[y.max()]],"black" )
    plt.show()
