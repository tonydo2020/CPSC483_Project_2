# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 00:45:00 2019

@author: shlakhanpal
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def distanceBetPointsAndCentroids(Y): # Y = [3, 300]
    for i in range(K): # 3
        for j in range(num_points): # 300, loops are getting distance between current datapoints & current centroids
            dist[i,j] = np.linalg.norm(initial_centroids[i] - X[j])
            Y[i,j] = dist[i,j]




def return_best(Y): # Y, best_value = [3, 300]
    best_value = np.zeros((K, num_points))  # Saves a 1 in that cell if it is lower value than its neighboring units

    min_values = (np.argmin(Y, axis = 0 )) # [1, 300]
    print(len(min_values))
    print(min_values)
    #best_value[min_values, :] = 1
    #best_value[row, column]

    avg_X = 0
    avg_Y = 0

    for i in range(K):
        print("i is: ", i)
        count = 0
        for j in range(num_points):
            if min_values[j] == i:
                avg_X = avg_X + X[j][0]
                avg_Y = avg_Y + X[j][1]
                count = count + 1
        avg_X = avg_X / count
        print(avg_X)
        avg_Y = avg_Y / count
        print(avg_Y)
        initial_centroids[i] = (avg_X, avg_Y)
        avg_X = 0
        avg_Y = 0
        count = 0

    print("initial centroid is now :" , initial_centroids)
    #print(best_value)

    return best_value


def display_centroids(min_values, X):
    a, b, c = np.zeros((300,2))
    a_count = 0
    b_count = 0
    c_count = 0
    for j in range(num_points):
        if min_values[j] == 0:
            a[a_count] = X[j]
            a_count = a_count + 1
        elif min_values[j] == 1:
            b[b_count] = X[j]
            b_count = b_count + 1
        elif min_values[j] == 2:
            c[c_count] = X[j]
            c_count = c_count + 1

    plt.scatter(a)
    plt.show()

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

initial_centroids = np.array([[3.0,3.0], [6.0,2.0], [8.0,5.0]])
print("initial_centroids is : ", initial_centroids)

plt.plot(X[:, 0], X[:, 1], 'go')
plt.plot(initial_centroids[:, 0], initial_centroids[:, 1], 'rx')
#plt.show()
#print("X first row ", X[0])

dist = np.zeros((K,num_points)) # rows by columns / centroid by data

Y = np.zeros((K,num_points)) # 3 by 300

min_values = (np.argmin(Y, axis = 0 )) # [1, 300]

for a in range(10):
    distanceBetPointsAndCentroids(Y)
#print("Y is: ", Y)
    best_value = return_best(Y)
#print("best array is: ", best_value)


display_centroids(min_values, X)

#plt.plot(X[:, 0], X[:, 1], 'go')
#plt.plot(initial_centroids[:, 0], initial_centroids[:, 1], 'rx')
#plt.show()




