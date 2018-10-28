#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from load_mnist import load_mnist
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
from sklearn.metrics import precision_recall_fscore_support
import time
import numpy as np
from sklearn.preprocessing import StandardScaler


# # Standardization
# Determine the mean and standard deviation for each feature, then subtract the mean from each feature,divide the values by corresponding standard deviation.
# \begin{equation}
# {x}' = \frac{x-\bar{x}}{\delta }
# \end{equation}

# In[ ]:



train_data, train_label = load_mnist("","train")
test_data,test_label = load_mnist("","t10k")

sc = StandardScaler()
X_main_std = sc.fit(train_data)

train_data = X_main_std.transform(train_data)
test_data = X_main_std.transform(test_data)


# # Cross Validation
# To choose appropriate max_depth, we divide the training set into 10 sections, train 10 times on it and each time use 9 sections as training data, 1 section as validation set, compare the results.

# In[ ]:


# Cross validation to find appropriate parameter(max depth)


Maxdepth = [2,4,6,10,15]
for maxdepth in Maxdepth:
    t1 = time.time()
    clf = tree.DecisionTreeClassifier(max_depth = maxdepth)
    scores = cross_val_score(clf, train_data, train_label, cv=10)
    t2 = time.time()
    total_time = t2-t1
    print("maxdepth is ",str(maxdepth),". Time:",str(total_time), "\nAverage Score:", sum(scores)/len(scores))


# # Train and Test
# We found max_depth = 15 would give us best result, we used it train model, used the model for predicting

# In[ ]:



maxdepth = 15

clf = tree.DecisionTreeClassifier(max_depth = maxdepth)
t1 = time.time()
clf.fit(train_data, train_label)
t2 = time.time()

pred=clf.predict(test_data)
acc_rate = np.sum(pred == test_label) / test_label.shape[0]
print("Time:",str(t2-t1),"seconds\n","Accuracy:",acc_rate)


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

