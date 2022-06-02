#-*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series

from sklearn.metrics import mean_squared_error , r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDConfig
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Subshape import SubshapeBuilder,SubshapeAligner,SubshapeObjects
from rdkit.Chem import rdFMCS

from bokeh.plotting import figure, output_file, show, ColumnDataSource, output_notebook
from bokeh.models import HoverTool, BoxSelectTool

class DrawMols():
    """
    Tool of drawing the molecule

    Parameters
    ----------
    path : str , file path of sdf file

    Sub functions
    -------
    show(self,index=[])
    save_img (self,index=[])
    common_substr (self,index=[])
    show_substr(self,substr_info)
    """
    def __init__(self,path):
        self.path = path

    def show(self,index=[]):
        """
        Draw the molecule of the user-specified index

        Parameters
        ----------
        index : list, specified index that user want to draw

        Returns
        -------
        Molecule images

        """
        pickimg = []
        cdk2mols = [x for x in Chem.SDMolSupplier(self.path,removeHs=False) if x is not None]
        for m in cdk2mols: tmp=AllChem.Compute2DCoords(m)
        for i in index :
            img = cdk2mols[i]
            pickimg.append(img)
        return Draw.MolsToGridImage(pickimg[:],molsPerRow=4)

    def save_img (self,index=[]) :
        """
        Save the molecule of the user-specified index as image files

        Parameters
        ----------
        index : list, specified index that user want to save

        Returns
        -------
        list, saved file names

        """
        imglist = []
        mols = [x for x in Chem.SDMolSupplier(self.path,removeHs=False) if x is not None]
        for m in mols: tmp=AllChem.Compute2DCoords(m)
        for i in index :
            moll = mols[i]
            Draw.MolToFile(moll,'%d.png'%(i))
            #img =  Draw.MolsToGridImage(moll)
            #img.save('%s%d.png'%(save_path,i))
            imglist.append('%d.png'%(i))
        return imglist

    def common_substr (self,index=[]) :
        """
        Finding common structure among specified index molecules

        Parameters
        ----------
        index : list, specified molecule's index

        Returns
        -------
        Smart string of common structure

        """
        pickmol = []
        mols = [x for x in Chem.SDMolSupplier(self.path,removeHs=False) if x is not None]
        for i in index :
            mol = mols[i]
            pickmol.append(mol)
        res=rdFMCS.FindMCS(pickmol)
        mol = Chem.MolFromSmarts(res.smartsString)
        return res.smartsString

    def show_substr(self,substr_info):
        """
        Draw the common structure using smart string

        Parameters
        ----------
        substr_info : str, return value of common_substr()

        Returns
        -------
        Image of common structure
        """
        return Chem.MolFromSmarts(substr_info)


def mol_plot(X_data, y_data, feature_set, imglist):
    """
    Draw the interactive prediction graphe with molecule images

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    y_data : pandas DataFrame , shape = (n_samples,)
    feature_set : list, set of features that make up model
    imglist : return value of save_mol()

    Returns
    -------
    Interactive prediction graphe
    """
    output_notebook()
    x = X_data.loc[:,feature_set].as_matrix()
    Ay = y_data.as_matrix()
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

    source = ColumnDataSource(
    data=dict(x=aay,y=ppy, imgs = imglist)
    )
    hover = HoverTool(
        tooltips="""
        <div>
            <div>
                <img
                    src="@imgs" height="100" alt="@imgs" width="180"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                    ></img>
                    </div>
                    <div>
                    <span style="font-size: 17px; font-weight: bold;">@desc</span>
                    <span style="font-size: 15px; color: #966;">[$index]</span>
                    </div>
                    <div>
                    <span style="font-size: 15px;">Location</span>
                    <span style="font-size: 10px; color: #696;">($x, $y)</span>
                    </div>
                    </div>
                    """
                    )

    p = figure(plot_width=600, plot_height=600, tools=[hover],
    title="Predict & Actual")
    p.circle('x', 'y', size=20, source=source, color="green", alpha=0.5 )
    show(p)
