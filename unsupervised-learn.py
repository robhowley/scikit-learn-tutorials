
# unsuperverised learning tutorial
# learning without the target labels



# K-means clustering
# split the obserations into separated groups (clusters)
# create clusters of roughly equal variance and w min inertia
# chooses centroids that min within cluster sos
# init='k-means++' provides alg with better starting point
# lots of ways for this alg to fail
  # how many clusters do we choose?
  # what is the starting point? stuck at local min for soln
from sklearn import cluster, datasets

iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X_iris)
 # KMeans(copy_x=True, init='k-means++', ...
print(k_means.labels_[::10])
 # [1 1 1 1 1 0 0 0 0 0 2 2 2 2 2]
print(y_iris[::10])
 # [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2]


# k-means example: vector quantization
import scipy as sp
import numpy as np
try:
  lena = sp.lena()
except AttributeError:
  from scipy import misc
  lena = misc.lena()
  
X = lena.reshape((-1, 1)) # We need an (n_sample, n_feature) array
k_means = cluster.KMeans(n_clusters=5, n_init=1)
k_means.fit(X)
 # KMeans(copy_x=True, init='k-means++', ...
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
lena_compressed = np.choose(labels, values)
lena_compressed.shape = lena.shape



# hierarchical agglomerative clustering: Ward
# hierarchical analysis builds a hierarchy of clusters
  # agglomerative: bottom up; successively merges observations together
  # divisive: top down; ill-posed
# Ward clustering minimizes a criterion similar to k-means in a bottom-up approach
  # for large cluster count, more efficient than k-means

# connectivity constrained: specify which samples should be clustered via adjacency mat
# connectivity example with pixels
lena = sp.misc.lena()
# Downsample the image by a factor of 4
lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
X = np.reshape(lena, (-1, 1))

# Define the structure A of the data.
# pixels connected to their neighbors.
from sklearn.feature_extraction.image import grid_to_graph
connectivity = grid_to_graph(*lena.shape)

# compute the ward cluster
import time
print("Compute structured hierarchical clustering...")
st = time.time()
n_clusters = 15  # number of regions
ward = cluster.Ward(n_clusters=n_clusters, connectivity=connectivity).fit(X)
label = np.reshape(ward.labels_, lena.shape)
print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", label.size)
print("Number of clusters: ", np.unique(label).size)



# feature agglomeration: merge similar features
# used when observation ct small relative to feature ct
digits = datasets.load_digits()
images = digits.images
X = np.reshape(images, (len(images), -1))
connectivity = grid_to_graph(*images[0].shape)

agglo = cluster.WardAgglomeration(connectivity=connectivity,n_clusters=32)
agglo.fit(X)
 # WardAgglomeration(compute_full_tree='auto',...

# dimension reducing estimators usually have a transform and inverse_transform
# method to generate the red dim approximations
X_reduced = agglo.transform(X)
X_approx = agglo.inverse_transform(X_reduced)
images_approx = np.reshape(X_approx, images.shape)



# decomposition methods
# want to express multivariate data X in a lower dim
# w loadings L and components C st X = LC

# PCA: selects orthogonal components that explain maximum variance
# when used to transform data PCA projects orig data on principal subspace
# Create a signal with only 2 useful dimensions
x1 = np.random.normal(size=100)
x2 = np.random.normal(size=100)
x3 = x1 + x2
X = np.c_[x1, x2, x3]

from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(X)
 # PCA(copy=True, n_components=None, whiten=False)
print(pca.explained_variance_)
 # [  2.18565811e+00   1.19346747e+00   8.43026679e-32]

# As we can see, only the 2 first components are useful
pca.n_components = 2
X_reduced = pca.fit_transform(X)
X_reduced.shape
 # (100, 2)

 
# ICA: independent components analysis
# distribution of loadings carries a max amount of independent information
# it is able to recover non-Gaussian independent signals
# Generate sample data
time = np.linspace(0, 10, 2000)
s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
S = np.c_[s1, s2]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise
S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1], [0.5, 2]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations (matrix mult)

# Compute ICA
ica = decomposition.FastICA()
S_ = ica.fit_transform(X)  # Get the estimated sources
A_ = ica.mixing_.T
np.allclose(X,  np.dot(S_, A_) + ica.mean_) # test elementwise (almost) equality
 # True