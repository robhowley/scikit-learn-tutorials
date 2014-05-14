# following along with the scikit-learn statistical inference tutorial
# tutorial homepage: http://scikit-learn.org/stable/tutorial/statistical_inference/index.html

from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
data.shape

digits = datasets.load_digits()
digits.images.shape   # (1797, 8, 8)
import pylab as pl
pl.imshow(digits.images[-1], cmap=pl.cm.gray_r)

# transform data
# data should be in (n_obs, n_features) shape
# the -1 means fill in whatever dimension value is necessary
data = digits.images.reshape((digits.images.shape[0], -1))
data.shape    # (1797, 64)
