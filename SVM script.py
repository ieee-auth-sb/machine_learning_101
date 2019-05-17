#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# source: https://www.kaggle.com/joniarroba/noshowappointments

# In[ ]:


# import dataset
data = pd.read_csv("KaggleV2-May-2016.csv")
print(data.shape)
print(data.head())


# In[ ]:


# split to x,y
x = data.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood', 'No-show'], axis=1)
# Set sex from string to int (M->0, F->1)
x['Gender'] = x['Gender'].astype('category')
x['Gender'] = x['Gender'].cat.reorder_categories(['M', 'F'], ordered=True)
x['Gender'] = x['Gender'].cat.codes
y = data['No-show']
# print formated dataset
x.head()


# In[ ]:


# print formated output
y.head()


# In[ ]:


# split to training & testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

# create model
model = SVC(kernel='linear')

# train model
model.fit(x_train, y_train)

# make predictions
y_pred = model.predict(x_test)

print(model)


# In[ ]:


# evaluate model
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

