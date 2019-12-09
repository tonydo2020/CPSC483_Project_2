# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 00:45:00 2019

@author: shlakhanpal
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def distanceBetPointsAndCentroids(Y):
    counter = 0
    for i in range(K): # 3
        for j in range(num_points): # 300, loops are getting distance between current datapoints & current centroids
            print("when centroid is: ", i)
            print("when point is at : ", j)
            dist[i,j] = np.linalg.norm(initial_centroids[i] - X[j])
            #print (dist[i,j])

            Y[i][j] = dist[i,j]
            counter = counter + 1


def return_best(Y):
    best = 0
    print("k is:" , K)
    print("num_points is: ", num_points)
    temp_array = np.zeros((300,1))
    for i in range(K - 1):
        for j in range(num_points):
            print("i is currently:" , i)
            print("j is currently:", j)
            if Y[i][j] <= Y[i + 1][j]:
                best = Y[i][j]
            else:
                best = Y[i + 1][j]
            temp_array[i][j] = best
            print("temp_array is: ", temp_array[i][j])
        best = 0
    return temp_array



datafile = 'kmeansdata.mat'
data = scipy.io.loadmat(datafile)


X = np.array([[1,1],[2,1],[4,3],[5,4]]) # class activity

X = data['X']
#print('Printing data in X...')

#print(X)

print('Dimensions of X', X.shape)

num_points = len(X)
#print("Total number of points in dataset, ie. X:", num_points)

K = 3 # class activity

initial_centroids = np.array([[3,3], [6,2], [8,5]])

plt.plot(X[:, 0], X[:, 1], 'go')
plt.plot(initial_centroids[:, 0], initial_centroids[:, 1], 'rx')

#print("X first row ", X[0])

dist = np.zeros((K,num_points))

Y = np.zeros((900,1))
Y_test = [[0,0,0]] * 300
print(Y_test)

print("Y.shape is", Y.shape)
distanceBetPointsAndCentroids(Y_test)
print("after distance: ", Y_test)

best_array = return_best(Y_test)
print("best array is: ", best_array)
#print(Y)
#print(Y[0][0])

plt.plot(X[:, 0], X[:, 1], 'go')
plt.plot(initial_centroids[:, 0], initial_centroids[:, 1], 'rx')





