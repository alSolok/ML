#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 01:11:44 2019

@author: diallo
"""

import numpy as np
import pandas as pds
import seaborn as sborn
import matplotlib.pyplot as mplt
import statsmodels as stat


# creation d'une variable qui stocke les dataframes importés
dtst = pds.read_csv('lycee.csv')

#decoupage des données

X = dtst.iloc[:,:-1].values
Y = dtst.iloc[:,-1].values

#constructionde l'echantillon de training et de l'echantillion de test

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#contruction du modele de regression simple
from sklearn.linear_model import LinearRegression
regresseur = LinearRegression()
regresseur.fit(X_train, Y_train)

#etablir une prediction
y_prediction = regresseur.predict(X_test)

#prediciton sur nouvel data

#regresseur.predict(180)

#visualisation des resultats de l'ensemble d'entrainement

mplt.scatter(X_train,Y_train, color = 'red')
mplt.plot(X_train,regresseur.predict(X_train), color ='green')
mplt.title('rendement note sur la minute de revision')
mplt.xlabel('minutes passés à reviser')
mplt.ylabel('note en %')
mplt.show()

mplt.scatter(X_test,Y_test, color = 'yellow')
mplt.plot(X_test,regresseur.predict(X_test), color ='black')
mplt.title('rendement note sur la minute de revision')
mplt.xlabel('minutes passés à reviser')
mplt.ylabel('note en %')
mplt.show()