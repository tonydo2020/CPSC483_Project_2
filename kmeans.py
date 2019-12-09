import scipy.io
import numpy as np
import matplotlib.pyplot as plt

datafile = 'kmeansdata.mat'
data = scipy.io.loadmat(datafile)

print('type of points array', type(data))


X = np.array([[3,3], [6,2], [8,5]])

print('Converting to numpy array, X', type(X))

X = data['X']

print("printing data in X...")
print(X)

print('Dimensions of X', X.shape)

num_points = len(X)
print("Total number of points in dataset, ie: X: ", num_points)

K = 3

initial_centroids = np.array([[3,3], [6,2], [8,5]])

