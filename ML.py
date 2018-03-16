# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:41:54 2018

@author: Jiazhen
"""

# split data into train and test
from sklearn.model_selection import train_test_split

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']
all_X = train[columns]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.20,random_state=0)


# train model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_X[columns], train_y["Survived"])

# measure 
from sklearn.metrics import accuracy_score
predictions = lr.predict(test_X)
accuracy = accuracy_score(test_y, predictions)
print(accuracy)

# cross validation
from sklearn.model_selection import cross_val_score
import numpy as np
lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_y, cv=10)
accuracy = np.mean(scores)
print(scores)
print(accuracy)



#Improving the features:
#   Feature Engineering: Create new features from the existing data.
#   Feature Selection: Select the most relevant features to reduce noise and overfitting.
#Improving the model:
#   Model Selection: Try a variety of models to improve performance.
#   Hyperparameter Optimization: Optimize the settings within each particular machine learning model.