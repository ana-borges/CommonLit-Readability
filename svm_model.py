import numpy as np
import pandas as pd
from sklearn.svm import SVR, NuSVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv('train_def.csv', encoding = 'latin-1')
test_df = pd.read_csv('test_def.csv', encoding = 'latin-1')

drop_feat = ['excerpt', 'cleaned_text', 'id', 'standard_error', 'target']

# Change to data_train and data_val when not using the full training dataset
X = train_df.drop(drop_feat, axis=1)
y = train_df['target']

### SVR models

# We leave out 'linear' and 'sigmoid' due to their bad results
svr_kernels = ['poly', 'rbf']
gamma = np.arange(0.1, 1.1, 0.3)

svr_pred =\
    [SVR(kernel=ker, C=100, gamma=gam, degree=3, epsilon=.1, coef0=1).fit(X, y).predict(X)\
     for ker in svr_kernels for gam in gamma]

svr_acc = [mean_squared_error(y, y_pred) for y_pred in svr_pred]

display([(ker, gam, acc) for ker in svr_kernels for gam in gamma for acc in svr_acc\
         if acc <= 0.01])

for ker, acc in list(zip(svr_kernels, svr_acc)):
    print(ker + ": " + str(acc))

### NuSVR models

nusvr_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

nusvr_pred =\
    [NuSVR(kernel=ker, C=100, gamma=0.1, degree=3, nu=.1, coef0=1).fit(X, y).predict(X)\
     for ker in nusvr_kernels]

nusvr_acc = [mean_squared_error(y, y_pred) for y_pred in nusvr_pred]

for ker, acc in list(zip(nusvr_kernels, nusvr_acc)):
    print(ker + ": " + str(acc))

## We can conclude that SVR works best
