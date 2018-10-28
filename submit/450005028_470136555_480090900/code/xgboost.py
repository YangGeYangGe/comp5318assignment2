#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# # Standardization
# Determine the mean and standard deviation for each feature, then subtract the mean from each feature,divide the values by corresponding standard deviation.
# \begin{equation}
# {x}' = \frac{x-\bar{x}}{\delta }
# \end{equation}

# In[ ]:


# Load data
train_data, train_label = load_mnist("","train")
test_data,test_label = load_mnist("","t10k")

# Do standardization on both training and test data
sc = StandardScaler()
X_main_std = sc.fit(train_data)
train_data = X_main_std.transform(train_data)
test_data = X_main_std.transform(test_data)


# # Set common parameters

# In[ ]:


# parameters for XGBoost
# will use all threads in default setting
param = {}
param['objective'] = 'multi:softmax'
param['eta'] = 0.3
param['num_class'] = 10
param["subsample"] = 0.8
param["colsample_bytree"] = 0.8
param["eval_metric"] = 'merror'
param['silent'] = 1

# number of trees
num_round = 100


# # Different tree method

# In[ ]:


# different tree method. 'approx' is faster than 'exact'. gpu_exact is a GPU method
param['tree_method'] = 'approx'
# param['tree_method'] = 'exact'
# param['tree_method'] = 'gpu_exact'


# # Cross Validation (Time consuming)
# To choose appropriate max_depth, we divide the training set into 10 sections, train 10 times on it and each time use 9 sections as training data, 1 section as validation set, compare the results.

# In[ ]:



def cv_test():
    kf = KFold(n_splits = 10)
    accs = []
    time1 = time.time()
    i = 0
    for train_index, val_index in kf.split(train_data):
        #divide original training data into new trainig data and validation data
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
        
        #predicting on validation data
        pred = bst.predict(dval)
        
        acc_rate = np.sum(pred == y_valid) / y_valid.shape[0]
        bst.__del__()
        accs.append(acc_rate)
        
    time2 = time.time()
    total_time = time2-time1
    print("maxdepth is ",str(max_depth),". Time:",str(total_time), "\nAverage Score:", sum(accs)/len(accs))

    
Maxdepth = [2,4,6,10,15]
for depth in Maxdepth:
    param['max_depth'] = depth
    cv_test()


# # Train and Test
# We found max_depth = 6 would give us best result, we used it train model, used the model for predicting

# In[ ]:


param['max_depth'] = 6
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


# # Confusion maxtrix

# In[ ]:


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
    else:
        pass
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
# Compute confusion matrix
cnf_matrix = confusion_matrix(pred, test_label)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=[0,1,2,3,4,5,6,7,8,9],
                      title='Confusion matrix, without normalization')
plt.show()


# # Precision, Recall, F1 score

# In[ ]:


precision, recall, f1, size = precision_recall_fscore_support(test_label,pred)
print(precision)
print(recall)
print(f1)

