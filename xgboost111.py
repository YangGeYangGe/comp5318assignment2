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




train_data, train_label = load_mnist("","train")
test_data,test_label = load_mnist("","t10k")

sc = StandardScaler()
X_main_std = sc.fit(train_data)

train_data = X_main_std.transform(train_data)
test_data = X_main_std.transform(test_data)
param = {}
param['tree_method'] = 'gpu_exact'
param['objective'] = 'multi:softmax'
param['eta'] = 0.3
param['num_class'] = 10
param["subsample"] = 0.8
param["colsample_bytree"] = 0.8
param["eval_metric"] = 'merror'
param['silent'] = 1


x_train, x_valid, y_train, y_valid = train_test_split(train_data, train_label, test_size=0.2)
dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_valid, label=y_valid)
dtest = xgb.DMatrix(test_data, label=test_label)


# times = []
# accs = []

# depths = [2,4,6,10,15]
# for i in depths:
num_round = 150
param['max_depth'] = 15

early_stopping = 5
evallist = [(dtrain, 'train'),(dval, 'validation')]
time1 = time.time()
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=early_stopping)
time2 = time.time()
pred = bst.predict(dtest)
acc_rate = np.sum(pred == test_label) / test_label.shape[0]
bst.__del__()
print('Test accuracy using softmax = {}'.format(acc_rate))
print((time2-time1)," seconds")

# times.append(time2-time1)
# accs.append(acc_rate)
# print("time is ",times)
# print("accuracy is ",accs)
# confusion_matrix(test_label, pred)
# Compute confusion matrix

cnf_matrix = confusion_matrix(test_label, pred)
print(cnf_matrix)
np.set_printoptions(precision=3)
print(precision_recall_fscore_support(test_label, pred))
# Plot non-normalized confusion matrix
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=[0,1,2,3,4,5,6,7,8,9],
                      title='Confusion matrix, without normalization')

plt.show()

