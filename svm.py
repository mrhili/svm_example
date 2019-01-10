# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 02:33:12 2019

@author: hp
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.txt')

df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

#print( df.head() )

X = np.array( df.drop(['class'],1) )
y = np.array( df['class'] )

X_train , X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2 )

clf = svm.SVC(n_jobs =-1)
clf.fit( X_train , y_train )

accuracy = clf.score(X_test, y_test)

#print( accuracy )

example_measures = np.array([2,7,10,10,7,10,4,9,4])
example_measures = example_measures.reshape(len( example_measures ),-1)
predection = clf.predict(example_measures)

print( predection )