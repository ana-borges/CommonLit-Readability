import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv('train_def.csv', encoding = 'latin-1')
test_df = pd.read_csv('test_def.csv', encoding = 'latin-1')

drop_feat = ['excerpt', 'cleaned_text', 'id', 'standard_error', 'target']

# Change to data_train and data_val when not using the full training dataset
X = train_df.drop(drop_feat, axis=1)
y = train_df['target']


svr_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

svr_pred =\
    [SVR(kernel=ker, C=100, gamma=0.1, degree=3, epsilon=.1, coef0=1).fit(X, y).predict(X)\
     for ker in svr_kernels]

svr_acc = [mean_squared_error(y, y_pred) for y_pred in svr_pred]

for ker, acc in list(zip(svr_kernels, svr_acc)):
    print(ker + ": " + str(acc))


