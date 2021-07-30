import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def cancer_prediction(features):
    dataset=pd.read_csv('Data.csv')
    dataset.head()
    dataset.isnull().sum()
    dataset.dtypes
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(dataset.drop(labels=['Class'],axis=1),dataset['Class'],test_size=0.2,random_state=0)
    x_train.head()
    y_train.head()
    from sklearn.feature_selection import mutual_info_classif
    mutual_info=mutual_info_classif(x_train,y_train)
    mutual_info
    mutual_info=pd.Series(mutual_info)
    mutual_info.index=x_train.columns
    mutual_info.sort_values(ascending=False)
    mutual_info.sort_values(ascending=False).plot.bar(figsize=(20,8))
    from sklearn.feature_selection import SelectKBest
    best=SelectKBest(mutual_info_classif,k=5)
    best.fit(x_train,y_train)
    x_train.columns[best.get_support()]
    x_train=x_train.drop(['Marginal Adhesion'],axis=1)
    x_train=x_train.drop(['Mitoses'],axis=1)
    x_train=x_train.drop(['Sample code number'],axis=1)
    x_test=x_test.drop(['Marginal Adhesion'],axis=1)
    x_test=x_test.drop(['Mitoses'],axis=1)
    x_test=x_test.drop(['Sample code number'],axis=1)
    from sklearn.ensemble import RandomForestClassifier
    classifier=RandomForestClassifier(n_estimators=51,criterion='entropy',random_state=0)
    classifier.fit(x_train,y_train)
    x_test=np.array(features)
    x_test=x_test.reshape((1,-1))

    return classifier.predict(x_test)[0]