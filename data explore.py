# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:28:40 2018

@author: Jiazhen
"""

df.shape()

df.describe()



# two histograms to compare visually
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()



# convert continuous to categorial
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,18,100]
label_names = ["Missing","Child","Adult"]

train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)

# Most machine learning algorithms can't understand text labels, so we have to convert our values into numbers
# While the class of each passenger certainly has some sort of ordered relationship,
# the relationship between each class is not the same as the relationship between the numbers 1, 2, and 3.
# For instance, class 2 isn't "worth" double what class 1 is, and class 3 isn't "worth" triple what class 1 is.
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

train = create_dummies(train,"Pclass")
test = create_dummies(test,"Pclass")