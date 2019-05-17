#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# source: https://www.kaggle.com/mirichoi0218/insurance

# In[2]:


#Importing data
df = pd.read_csv('insurance.csv')
df.info()
df.head()


# In[3]:


#Setting sex from string to int (male->0, female->1)
df['sex'] = df['sex'].astype('category')
df['sex'] = df['sex'].cat.reorder_categories(['male', 'female'], ordered=True)
df['sex'] = df['sex'].cat.codes
#Setting smoker from string to int (no->0, yes->1)
df['smoker'] = df['smoker'].astype('category')
df['smoker'] = df['smoker'].cat.reorder_categories(['no', 'yes'], ordered=True)
df['smoker'] = df['smoker'].cat.codes
#Deleting 'region' column
df.drop('region', axis=1, inplace = True)

df.info()
df.head()


# In[4]:


#Exporting data to arrays x, y
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
#Spliting data 
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.25)


# In[5]:


#Least squares model
lr_model = linear_model.LinearRegression()
lr_model.fit(x_train, y_train)
print('LINEAR MODEL')
print('y =' , float(lr_model.coef_[0]) , '* x1 +' , float(lr_model.coef_[1]) , '* x2 +' ,      float(lr_model.coef_[2]) , '* x3 +' , float(lr_model.coef_[3]) , '* x4 +\n' ,      float(lr_model.coef_[4]) , '* x5 +' , float(lr_model.intercept_))
print('Training score: {}'.format(lr_model.score(x_train, y_train)))
print('Test score: {}'.format(lr_model.score(x_test, y_test)))
y_pred = lr_model.predict(x_test)


# In[6]:


#Least squares error
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print('RMSE: {}'.format(rmse))


# In[7]:


x_request = np.array([28,0,33.0,3,0])
print(lr_model.predict(x_request.reshape(1,-1)))

