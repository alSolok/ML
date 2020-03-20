#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:47:39 2019

@author: diallo
"""

import statsmodels as stat
import seaborn as sborn
import pandas as pds
import matplotlib.pyplot as mplt
import numpy as np

# creation d'une variable qui stocke les dataframes importés

dtst = pds.read_csv('credit_immo.csv')

#decoupage des données

x = dtst.iloc[:,-9:-1].values
y = dtst.iloc[:,-1].values

#data cleaning

from sklearn.preprocessing import Imputer

#clean values null
imptr = Imputer(missing_values="NaN", strategy= 'mean', axis=0)

#target the column
imptr.fit(x[:,0:1])
imptr.fit(x[:,7:8])

#transform the value null in mean
x[:,0:1] = imptr.transform(x[:,0:1])
x[:,7:8] = imptr.transform(x[:,7:8])

#Données catégoriques

#codage des variable indépendante

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labEncr_x = LabelEncoder()
x[:,2] = labEncr_x.fit_transform(x[:,2])
x[:,5] = labEncr_x.fit_transform(x[:,5])
onehotEncr = OneHotEncoder(categorical_features=[2])
onehotEncr = OneHotEncoder(categorical_features=[5])
x = onehotEncr.fit_transform(x).toarray()

#variable dependante

labEncr_y = LabelEncoder()
y = labEncr_y.fit_transform(y)

#fractionner l'ensemble des données

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#normalisation de données

from sklearn.preprocessing import normalize
x_train = normalize(x_train)
x_test = normalize(x_test)

from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
x_train = Sc.fit_transform(x_train)
x_test = Sc.fit_transform(x_test)