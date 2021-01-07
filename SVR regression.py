# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:51:23 2020

@author: abhit
"""

import numpy as np # linear algebra
import pandas as pd # data processing

import math,datetime
import time
import arrow
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,svm # Preprocessing for scaling data,Accuracy,Processing speed ,cross validation for training and testing
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


df = pd.read_csv(r'D:\VIT\SDP\ABC_data.csv')
print (df.head())

print(df.columns)

df=df[['open','high','low','close','volume']]
print (df.head())

df['HIGHLOW_PCT']=(df['high']-df['close'])/(df['close'])*100
#Calculating new and old prices
df['PCT_Change']=(df['close']-df['open'])/(df['open'])*100
# Extracting required data from file
df=df[['close','HIGHLOW_PCT','PCT_Change','volume']]
print (df.head())

df['HIGHLOW_PCT'].plot(c='r')
df['PCT_Change'].plot(c='b')
plt.show()

forecast_col='close'

df.fillna(-99999,inplace=True)

forecast_out=13
print (forecast_out)
df['label']=df[forecast_col].shift(-forecast_out)
print (df.head())

X = np.array(df.drop(['label'],1))

X = preprocessing.scale(X)

X = X[:-forecast_out] #taking away last data used for forecast

X_lately=X[-forecast_out:] #collecting the forecast data


df.dropna(inplace=True)
y=np.array(df['label'])
print (len(X),len(y))


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
clf=svm.SVR()
clf.fit(X_train,y_train)


accuracy=clf.score(X_test,y_test)
print ("Accuracy of prediction is:",accuracy)

forecast_set=clf.predict(X_lately) #predicting the forecast


#print (forecast_set,accuracy,forecast_out)

df['Forecast']=np.nan
print(df.head())
last_date=df.iloc[-1].name
last_unix = arrow.get(last_date).timestamp
one_day=86400
next_unix=last_unix + one_day

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix +=one_day
    df.loc[next_date]=[np.nan for j in range(len(df.columns)-1)] + [i]
 



df['close'].plot(c='g')
df['Forecast'].plot(c='b')
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')

plt.show()