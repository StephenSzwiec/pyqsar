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
        x = self.X_data.loc[:,self.feature_set].values
        y = self.y_data.values
        pred_plotY = np.zeros_like(y)
        g_mlrr = LinearRegression()
        g_mlrr.fit(x,y)
        pred_plotY = g_mlrr.predict(x)
        plt.ylabel("Predicted Y")
        plt.xlabel("Actual Y")
        plt.scatter(y,pred_plotY,color=['gray'])
        plt.plot([y.min() , y.max()] , [[y.min()],[y.max()]],"black" )
        plt.show()

    def williams_plot(self, exdataX=None, exdataY=None) :
        """
        Show residuals of training plot
        Optionally, show residuals of test plot

        Returns
        -------
        None
        """
        if (exdataX or exdataY) :
            print("Please input both X and Y data")
            return
        test_set = False
        x = self.X_data.loc[:,self.feature_set].values
        y = self.y_data.values
        g_mlrr = LinearRegression()
        g_mlrr.fit(x, y)
        std = np.std(y)
        Y_pred = g_mlrr.predict(x)
        H = Hin = (Y_pred / std) * ( x * np.linalg.pinv(x).T).sum(1)
        residuals = res = (Y_pred - y) / std
        if (exdataX != None and exdataY != None) :
            test_set = True
            xex = exdataX.loc[:self.feature_set].values
            yex = exdataY.values
            Y_pred_ex = g_mlrr.predict(xex)
            residuals_ex = (Y_pred_ex - yex) / std
            Hex = ( Y_pred_ex / std )  * ( xex * np.linalg.pinv(xex).T).sum(1)
            H.append(Hex)   # append Hex to H
            residuals.append(residuals_ex)  # append residuals of test data
        hii = 3 * ( (len(self.feature_set) + 1) / len(Y_pred) )
        H_min = min(H-0.1)
        plt.axline(xy1=(H_min,0),slope=0)
        plt.axline(xy1=(H_min,3),slope=0,linestyle="--")
        plt.axline(xy1=(H_min,-3),slope=0,linestyle="--")
        plt.axline(xy1=(hii, -3.5), xy2=(hii, 3.5))
        plt.ylabel("Std. Residuals")
        plt.xlabel("Hat Values")
        plt.ylim([-3.5,3.5])
        plt.scatter(Hin,res,color=['gray'])
        if(test_set) :
            plt.scatter(Hex,residuals_ex,color=['red'])
        plt.plot()
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
        x = self.X_data.loc[:,self.feature_set].values
        Ay = self.y_data.values
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
        x = self.X_data.loc[:,self.feature_set].values
        y = self.y_data.values
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
    Prediction external data set

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
    x = X_data.loc[:,feature_set].values
    y = y_data.values
    exd = exdataX.loc[:,feature_set].values
    exdY = exdataY.values

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
