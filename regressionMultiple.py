#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:40:02 2019

@author: diallo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as mplt 

dtst = pd.read_csv('EU_I_PIB.csv')
X = dtst.iloc[:, -4:].values
Y = dtst.iloc[:,-5].values

#gerer la dummy

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncr_X = LabelEncoder()
X[:,0] = labelEncr_X.fit_transform(X[:,0])
onehotEnc_X = OneHotEncoder(categorical_features= [0])
X = onehotEnc_X.fit_transform(X).toarray()

#division de l'echantillon

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#creation de notre model de regression multiple

from sklearn.linear_model import LinearRegression
regresseur = LinearRegression()
regresseur.fit(X_train,Y_train)

#faire une prediction

y_prediction = regresseur.predict(X_test)

regresseur.predict(np.array([[0,1,587923,48562,589654]]))