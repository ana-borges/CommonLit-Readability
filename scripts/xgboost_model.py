import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv('train_def.csv', encoding = 'latin-1')
test_df = pd.read_csv('test_def.csv', encoding = 'latin-1')

data_train, data_val, target_train, target_val = \
    train_test_split(train_df, train_df["target"], test_size=0.3, random_state=5)

drop_feat = ['excerpt', 'cleaned_text', 'id', 'standard_error', 'target']

# Change to data_train and data_val when not using the full training dataset
X_train = data_train.drop(drop_feat, axis=1)
X_val = data_val.drop(drop_feat, axis=1)

#Classifier
#############################################
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import Ridge

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
#from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LinearRegression

# pipe = Pipeline([
#     ('tfid', TfidfVectorizer(ngram_range=(1,2))),
#     ('model', xgb.XGBRegressor(
#         learning_rate=0.1,
#         max_depth=7,
#         n_estimators=80,
#         use_label_encoder=False,
#         eval_metric='rmse',
#     ))
# ])


# pipe.fit(np.array(X_train), np.array(target_train))
foo = XGBRegressor()
foo.fit(X_train, target_train)

kfold = KFold(n_splits=5, random_state=7, shuffle=True)
results = cross_val_score(foo, X_train, target_train, cv=kfold)

y_test_pred = foo.predict(X_val)
mse = mean_squared_error(y_test_pred, target_val)
print(mse)

# Fit the pipeline with the data
# pipe.fit(X_train, data_val)

# reg = Ridge(alpha=.1)
# reg.fit(X_train, target_train)
# predictions = reg.predict(X_val)
