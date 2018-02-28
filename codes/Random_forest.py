# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:50:48 2018

@author: Arafat
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd

def k_fold(model, x, y):
    scr = cross_val_score(model, x, y, cv=10, scoring='accuracy')
    return scr.mean()

    
file = 'first.xlsx'
    
# Load spreadsheet
data = pd.read_excel(file)
    
#columns
fcols = [
             'competitive programming background',
             'professional skill',
             'research background',
             'final year projct type',
             'enthusiasm',
             'teamwork ability',
             'communication & network skill',
             'cgpa'
]
    
    #selecting train data
x1 = data[fcols]
y1 = data['current job field']
    
iris = load_iris()
x = iris.data
y = iris.target

accu = list()
for i in range(10, 501, 10):
    rnd = RandomForestClassifier(n_estimators = i)
    accu.append(k_fold(rnd, x, y))

x_range = range(1, 991)
plt.plot(x_range, accu)
plt.show()
