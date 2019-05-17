import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# import dataset
data = pd.read_csv("bill_authentication.csv")
print(data.shape)
print(data.head())

# split to x,y
x = data.drop('Class', axis=1)
y = data['Class']

# split to training & testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

# create model
model = SVC(kernel='linear')

# train model
model.fit(x_train, y_train)

# make predictions
y_pred = model.predict(x_test)

# evaluate model
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
