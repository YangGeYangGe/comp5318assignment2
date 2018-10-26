from sklearn.model_selection import KFold
from load_mnist import load_mnist
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import precision_recall_fscore_support

train_data, train_label = load_mnist("","train")
test_data,test_label = load_mnist("","t10k")

sc = StandardScaler()
X_main_std = sc.fit(train_data)

train_data = X_main_std.transform(train_data)
test_data = X_main_std.transform(test_data)


sc = StandardScaler()
X_main_std = sc.fit(train_data)

train_data = X_main_std.transform(train_data)
test_data = X_main_std.transform(test_data)

param = {}
param['tree_method'] = 'approx'
# param['tree_method'] = 'exact'
param['objective'] = 'multi:softmax'
param['eta'] = 0.3
param['num_class'] = 10
param["subsample"] = 0.8
param["colsample_bytree"] = 0.8
param["eval_metric"] = 'merror'
param['silent'] = 1
num_round = 100

# [1,3,5,7,9]
param['max_depth'] = 5

import time
kf = KFold(n_splits = 10)
dtest = xgb.DMatrix(test_data, label=test_label)
accs = []
precisions = np.zeros((10,))
recalls = np.zeros((10,))
f1s = np.zeros((10,))

time1 = time.time()
i = 0
for train_index, val_index in kf.split(train_data):
    x_train = train_data[train_index]
    y_train = train_label[train_index]
    x_valid = train_data[val_index]
    y_valid = train_label[val_index]
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_valid, label=y_valid)
    evallist = [(dtrain, 'train'),(dval, 'validation')]
    
    #training
    early_stopping = 5
    
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=early_stopping,verbose_eval=False)
    
    #predicting
    pred = bst.predict(dtest)
    acc_rate = np.sum(pred == test_label) / test_label.shape[0]
    
    bst.__del__()
    
    accs.append(acc_rate)
    pr, re, f1, size = precision_recall_fscore_support(test_label, pred)
    
#     precisions.append(pr)
#     recalls.append(re)
#     f1s.append(f1)
    precisions += pr
    recalls += re
    f1s += f1

    print(i)
    i += 1
#     print(pr.shape)

time2 = time.time()
total_time = time2-time1

precisions /= 10
recalls /= 10
f1s /= 10

print(total_time)
print(sum(accs)/10)
print(precisions)
print(recalls)
print(f1s)

