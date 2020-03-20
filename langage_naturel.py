#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 07:50:25 2019

@author: diallo
"""

import pandas as pd
import matplotlib.pyplot as mplt
import numpy as np

dtst = pd.read_csv('NLP_MLP_last.csv')
#dtst.head()
#dtst.columns

from collections import Counter
titres = ["GSS 2.0: CSS polyfills from the future",
"Discover what your friends really think of you",
"The GNU Make Book Early Access",
"SqueezeMail MS Outlook addin",
"The Copenhagen Wheel transforms your bicycle into a smart electric hybrid"]
#trouvez les mots unique de ces titres

mot_uniques = list(set(" ".join(titres).split(" ")))
def make_matrix(titres, vocab):
    matrix = []
    for titre in titres:
        #comptez chaque mot dans le titre et faites un dictionnaire
        counter = Counter(titre)
        #transformer le dictionnaire en une ligne matricielle en utilisant vocab
        row = [counter.get(w, 0) for w in vocab]
        matrix.append(row)
    df = pd.DataFrame(matrix)
    df.columns = mot_uniques
    return df

print(make_matrix(titres,mot_uniques))

import re

nv_titres = [re.sub(r'[^\w\s\d]', '', h.lower()) for h in titres]
nv_titres = [re.sub("\s+", " ",h) for h in nv_titres]

mot_uniques = list(set(" ".join(nv_titres).split(" ")))

print(make_matrix(titres,mot_uniques))

#lisez et divisez le fichier des mots vides

with open("stop_words.txt", 'r') as f:
    stopwords = f.read().split("\n")

#faire le meme remplacement de ponctuation que nous avons fait pour les titres,
#donc nous comparons les bonnes choses
    
stopwords = [re.sub(r'[^\w\s\d]', '', s.lower()) for s in stopwords]
mot_uniques = list(set(" ".join(nv_titres).split(" ")))

#suppression des mots vides du vocabulaire

mot_uniques = [w for w in mot_uniques if w not in stopwords]

print(make_matrix(nv_titres,mot_uniques))

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(lowercase = True, stop_words ="english")

matrix = vectorizer.fit_transform(titres)

print(matrix.todense())


dtst['full_test'] = dtst['titre'] + " "+ dtst['url']

full_matrix = vectorizer.fit_transform(dtst['titre'].values.astype('U'))
print(full_matrix.shape)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(full_matrix, dtst['approbation'], test_size = 0.2, random_state = 0)

from sklearn.linear_model import Ridge

reg = Ridge(alpha = 1)

reg.fit(X_train,Y_train)

predictions = reg.predict(X_test)