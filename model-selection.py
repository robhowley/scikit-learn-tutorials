
# model selection tutorial

from sklearn import datasets, svm

# load digits data set, fit support vector machine model, compute fit score
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])
# 0.97999999999999998

# perform a kfolds validation the hard way
import numpy as np
X_folds = np.array_split(X_digits, 3) # split data into 3 groups
y_folds = np.array_split(y_digits, 3)

scores = list()
for k in range(3):
  # We use 'list' to copy, in order to 'pop' later on
  X_train = list(X_folds)
  
  # remove kth group, use as test set
  # keep remaining sets as training (concatenate them into one)
  X_test  = X_train.pop(k)
  X_train = np.concatenate(X_train)
  y_train = list(y_folds)
  y_test  = y_train.pop(k)
  y_train = np.concatenate(y_train)
  
  # compute the fit score with kth group as test set
  scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
  
print(scores)
# [0.93489148580968284, 0.95659432387312182, 0.93989983305509184]


# cross validation generators
# available generators:
  # KFold(n,k): Split it K folds, train on K-1 and then test on left-out
  # StratifiedKFold(y,n,k): preserve class ratios / label distribution within each fold.
  # LeaveOneOut(n): Leave one observation out
  # LeaveOneLabelOut(labels): Takes a label array to group observations
from sklearn import cross_validation

# creates n_folds pairs of index lists in form ([train_indices], [test_indices])
k_fold = cross_validation.KFold(n=6, n_folds=3, indices=True)
for train_indices, test_indices in k_fold:
  print 'Train: %s | test: %s' % (train_indices, test_indices)
# Train: [2 3 4 5] | test: [0 1]
# Train: [0 1 4 5] | test: [2 3]
# Train: [0 1 2 3] | test: [4 5]

# use KFold to fit and test svc model
kfold = cross_validation.KFold(len(X_digits), n_folds=3)
# report back the score for each hold out set
[svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]) for train, test in kfold]
# [0.93489148580968284, 0.95659432387312182, 0.93989983305509184]

# can do this in one shot with helper function
# n_jobs=-1 means that the computation will be dispatched on all the CPUs of the computer.
cross_validation.cross_val_score(svc, X_digits, y_digits, cv=kfold, n_jobs=-1)
# array([ 0.93489149,  0.95659432,  0.93989983])


# Exercise: compute cross validation score w digits dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)

scores = []
kfold = cross_validation.KFold(n=len(Y))
for c in C_s:
  svc.C = c
  scores.append(np.mean(cross_validation.cross_val_score(svc, X, y, cv=kfold)))
  
print scores