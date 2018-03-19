# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:28:40 2018

@author: Jiazhen
"""

df.shape()

train[columns].describe(include='all',percentiles=[])



# two histograms to compare visually
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()

