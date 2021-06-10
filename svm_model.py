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
gamma = np.arange(0.1, 1.1, 0.3)
#epsilon = gamma.copy()

accuracies = [(ker, gam,\
             SVR(kernel=ker, C=100, gamma=gam, epsilon=.1).fit(X, y).predict(X))\
            for ker in svr_kernels for gam in gamma]

foo = [(ker, gam, eps) for ker in svr_kernels for gam in gamma for eps in epsilon]

accuracies = [mean_squared_error(machine.fit(X)) for X in machines]

# Fit regression model
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma=0.1, degree=3, epsilon=.1, coef0=1)

lw = 2

svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ['RBF', 'Linear', 'Polynomial']


results = [model.fit(X, y).predict(X) for model in svrs]

accuracies = [mean_squared_error(y, y_pred) for y_pred in results]

for name, acc in list(zip(kernel_label, accuracies)):
    print(name + ": " + str(acc))
