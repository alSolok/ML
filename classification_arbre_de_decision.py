#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 23:48:40 2019

@author: diallo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as mplt
import seaborn as sbrn
import statsmodels as stat

dtst = pd.read_csv('Social.csv')
X = dtst.iloc[:, [2,3]].values
Y = dtst.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.30, random_state = 0)

#Mise à l'cehelle
from sklearn.preprocessing import StandardScaler
stdSc = StandardScaler()
X_train = stdSc.fit_transform(X_train)
X_test = stdSc.fit_transform(X_test)
 
 #Construction du model d'arbre de décision
 
from sklearn.tree import DecisionTreeClassifier
classification = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classification.fit(X_train, Y_train)
 
 #Mise en place de la prédiction
 
y_prediction = classification.predict(X_test)
 
 #matrice de confusion
 
from sklearn.metrics import confusion_matrix
 
mat_cof = confusion_matrix(Y_test,y_prediction)
 
 #visualisation de donnée
 
from matplotlib.colors import ListedColormap
X_set,Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                      np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
mplt.contourf(X1, X2, classification.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
               alpha = 0.75, mat_confap = ListedColormap(('red','green')))
mplt.xlim(X1.min(), X1.max())
mplt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    mplt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                  c = ListedColormap(('red', 'black'))(i), label = j)
mplt.title('classification de l\'arbre de décision (training set)')
mplt.xlabel('Age')
mplt.ylabel('Salaire estimé')
mplt.legend()
mplt.show()