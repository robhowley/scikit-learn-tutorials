
# supervised learning tutorial

# have observed data matrix X and mapping it to (usually 1D) y
# All supervised estimators in the scikit-learn implement a fit(X, y)
# method to fit the model and a predict(X) method

# classification: goal is to "name" the observed data (y is vec of ints)
# regression: goal is to predict some continuous target

import numpy as np
from sklearn import datasets

# note objects representing estimators have methods
# estimator.fit(x_features,y_targets)
# estimator.predict(x_features)

# k-nearest neighbor classifier
# given a new observation X_test, find in the training set
# (i.e. the data used to train the estimator) the observation
# with the closest feature vector

# nearest neighbor example with iris data
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# y data is 3 different iris types
# vector contains a series of identifiers by 0, 1, or 2
# np.unique() selects the unique values in the vector
# returns them as an array
np.unique(iris_y)   # array([0, 1, 2])


# should always split data into training and test sets
np.random.seed(0)

# random permutation of integers [0,len(X)-1] for indices
indices = np.random.permutation(len(iris_X))

# all but last 10 indices are for training set
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]

# last 10 for test set
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
  # default settings of the knn classifier is
  # KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
  #    n_neighbors=5, p=2, weights='uniform')
knn.predict(iris_X_test)  # returns array of classifiers 0,1,2
iris_y_test               # compare to actual

# effectiveness of this algorithm quickly diminishes with dimension



# regression example with diabetes data
diabetes = datasets.load_diabetes()

# train model with all but last 20 datapoints
# last 20 are for testing
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

# fit the linear model object
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
  # LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
print(regr.coef_)
 #[   0.30349955 -237.63931533  510.53060544  327.73698041 -814.13170937
 #  492.81458798  102.84845219  184.60648906  743.51961675   76.09517222]

# The mean square prediction error from test data = mean( (y_test - yhat_test)^2 )
np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2) # 2004.56760268...

# Explained variance score (of test data): 1 is perfect prediction and 0
# means that there is no linear relationship between X and Y.
regr.score(diabetes_X_test, diabetes_y_test) # 0.5850753022690



# shrinkage
# good use case: when feature count is high and data count low
X = np.c_[ .5, 1].T
y = [.5, 1]
test = np.c_[ 0, 2].T
regr = linear_model.LinearRegression()

import pylab as pl
pl.figure()

np.random.seed(0)
for _ in range(6):
  this_X = .1*np.random.normal(size=(2, 1)) + X
  regr.fit(this_X, y)
  pl.plot(test, regr.predict(test))
  pl.scatter(this_X, y, s=3)
  
# ridge regression
# type of shrinkage estimator, shrinks regr coeffs toward zero
# example of bias/variance trade off
# as ridge param increases, so does bias, but variance decreases
regr = linear_model.Ridge(alpha=.1)

pl.figure()
np.random.seed(0)
for _ in range(6):
  this_X = .1*np.random.normal(size=(2, 1)) + X
  regr.fit(this_X, y)
  pl.plot(test, regr.predict(test))
  pl.scatter(this_X, y, s=3)
  
# can test different alphas
alphas = np.logspace(-4, -1, 6)

# for each alpha
#   1: set the alpha
#   2: fit the model with training data
#   3: compute the score with the test data
print([regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train).score(diabetes_X_test, diabetes_y_test) for alpha in alphas])
# [0.5851110683883, 0.5852073015444, 0.5854677540698, 0.5855512036503, 0.5830717085554, 0.57058999437]


# lasso: will set some coefficients to zero
# ridge only makes them small
# lasso is a sparse method. if dataset is small use LassoLars
regr = linear_model.Lasso()

# compute scores for each alpha as done above
scores = [regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train).score(diabetes_X_test, diabetes_y_test) for alpha in alphas]

# find alpha that yields best score
best_alpha = alphas[scores.index(max(scores))]

# refit model with "best" alpha
regr.alpha = best_alpha
regr.fit(diabetes_X_train, diabetes_y_train)
# Lasso(alpha=0.025118864315095794, copy_X=True, fit_intercept=True,
#   max_iter=1000, normalize=False, positive=False, precompute='auto',
#   tol=0.0001, warm_start=False)
print(regr.coef_)
# [   0.         -212.43764548  517.19478111  313.77959962 -160.8303982    -0.
#  -187.19554705   69.38229038  508.66011217   71.84239008]




# logistic regression: a linear model used for classification
# parameter settings
  # C parameter controls the amount of regularization
  # large C means less regularized
  # penalty=”l2” gives Shrinkage (i.e. non-sparse coefficients)
  # penalty=”l1” gives Sparsity.
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(iris_X_train, iris_y_train)
# LogisticRegression(C=100000.0, class_weight=None, dual=False,
#          fit_intercept=True, intercept_scaling=1, penalty='l2',
#          random_state=None, tol=0.0001)

# the input is multi class in single vector
# the object converts it into a Y target matrix
# predictions are converted back to the integer classifiers
logistic.predict(iris_X_test)

# classifier corresponds to largest predicted probability
pred_prob = logistic.predict_proba(iris_X_test)     # all predicted probabilities
pred_c = [list(p).index(max(p)) for p in pred_prob] # same pred as logistic.predict()


# support vector machines
# finds a combination of samples to build a plane maximizing the
# margin between the two classes
# normalized data is a good idea for svm
# param C is for regularization
  # a small value for C means the margin is calculated using many or all
  # of the observations around the separating line (more regularization)
  # a large value for C means the margin is calculated on observations
  # close to the separating line (less regularization)
  
from sklearn import svm
svc = svm.SVC(kernel='linear')
svc.fit(iris_X_train, iris_y_train)
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
#   kernel='linear', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)

# can use "kernel trick" for polynomial separation
# degree: polynomial degree
svc = svm.SVC(kernel='poly', degree=3)

svc = svm.SVC(kernel='rbf')
# gamma: inverse of size of
# radial kernel