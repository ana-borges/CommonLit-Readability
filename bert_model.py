# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"]="0";

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import ktrain
from ktrain import text

import pandas as pd
import numpy as np

train_df = pd.read_csv('train_def.csv', encoding = 'latin-1')
test_df = pd.read_csv('test_def.csv', encoding = 'latin-1')

data_train, data_val, target_train, target_val = \
    train_test_split(train_df, train_df["target"], test_size=0.3, random_state=5)

drop_feat = ['excerpt', 'cleaned_text', 'id', 'standard_error', 'target']

# Change to data_train and data_val when not using the full training dataset
X_train = data_train.drop(drop_feat, axis=1)
X_val = data_val.drop(drop_feat, axis=1)


x_train = np.array(data_train.cleaned_text)
y_train = np.array(target_train)
x_test  = np.array(data_val.cleaned_text)
y_test  = np.array(target_val)


trn, val, preproc = text.texts_from_array(x_train=x_train, y_train=y_train,
                                          x_test=x_test, y_test=y_test,
                                          ngram_range=3,
                                          maxlen=200,
                                          max_features=35000,
                                          preprocess_mode='bert')

bert = text.text_regression_model('bert', train_data=trn, preproc=preproc)
bert_learner = ktrain.get_learner(bert, train_data=trn, val_data=val, batch_size=6)

# Find the learning rate
bert_learner.lr_find() #gives error for some reason

bert_learner.lr_plot() #if bert_learner.lr_find() doesn't work this won't either

bert_learner.fit_onecycle(0.03, 10)

