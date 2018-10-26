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

def cv_test(max_depth = 2):
    kf = KFold(n_splits = 10)
    accs = []
    # all_train = xgb.DMatrix(train_data, label=train_label)
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
        pred = bst.predict(dval)
        acc_rate = np.sum(pred == y_valid) / y_valid.shape[0]
        bst.__del__()
        accs.append(acc_rate)
        print(i)
        i += 1
    time2 = time.time()
    total_time = time2-time1
    print(total_time)
    print(sum(accs)/10)



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

# [2,4,6,10,15]
depth = 6
param['max_depth'] = depth

# cv_test(depth)

dtest = xgb.DMatrix(test_data, label=test_label)
x_train, x_valid, y_train, y_valid = train_test_split(train_data, train_label, test_size=0.2)
dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_valid, label=y_valid)

evallist = [(dtrain, 'train'),(dval, 'validation')]

early_stopping = 5
time1 = time.time()
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=early_stopping)
pred = bst.predict(dtest)
time2 = time.time()

acc_rate = np.sum(pred == test_label) / test_label.shape[0]
print('Test accuracy using softmax = {}'.format(acc_rate))
runtime = time2-time1
print(runtime," seconds")