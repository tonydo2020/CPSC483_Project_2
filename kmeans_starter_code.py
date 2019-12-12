# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 00:45:00 2019

@author: shlakhanpal
"""

#CPSC 483
#Tony Do
#Sami Halwani

import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def distanceBetPointsAndCentroids(Y): # Y = [3, 300]
    for i in range(K): # 3
        for j in range(num_points): # 300, loops are getting distance between current datapoints & current centroids
            dist[i,j] = np.linalg.norm(initial_centroids[i] - X[j])
            Y[i,j] = dist[i,j]

def return_closest(Y):
    min_values = (np.argmin(Y, axis = 0 ))
    return min_values

def return_best(Y): # Y, best_value = [3, 300]
    min_values = (np.argmin(Y, axis = 0 )) # [1, 300]
    avg_X = 0
    avg_Y = 0
    for i in range(K):
        count = 0
        for j in range(num_points):
            if min_values[j] == i:
                avg_X = avg_X + X[j][0]
                avg_Y = avg_Y + X[j][1]
                count = count + 1
        avg_X = avg_X / count
        #print(avg_X)
        avg_Y = avg_Y / count
        #print(avg_Y)
        initial_centroids[i] = (avg_X, avg_Y)
        avg_X = 0
        avg_Y = 0
        count = 0

    #print("initial centroid is now :" , initial_centroids)
    #print(best_value)

    return initial_centroids
    
def logic(data,Y):
    fig, graph = plt.subplots(3)
    fig.suptitle("Project 2")
    X = data['X']
    x, y = zip(*X)
    
    graph[0].scatter(x,y, c = 'b')
    graph[0].plot(initial_centroids[0:, 0], initial_centroids[0:, 1], 'yx')
    graph[0].plot(initial_centroids[1:, 0], initial_centroids[1:, 1], 'rx')
    graph[0].plot(initial_centroids[2:, 0], initial_centroids[2:, 1], 'cx')

    distanceBetPointsAndCentroids(Y)
    initial_iteration_centroids = return_best(Y)
    print("at iteration 0 centroid values are: ", initial_iteration_centroids)
    min_values = return_closest(Y)

    for a in range(len(min_values)):
        if min_values[a] == 0:
            graph[1].scatter(x[a],y[a], c = 'y')
        elif min_values[a] == 1:
            graph[1].scatter(x[a],y[a], c = 'r')
        elif min_values[a] == 2:
            graph[1].scatter(x[a],y[a], c = 'c')
    
    graph[1].plot(initial_iteration_centroids[0:, 0], initial_iteration_centroids[0:, 1], 'yx')
    graph[1].plot(initial_iteration_centroids[1:, 0], initial_iteration_centroids[1:, 1], 'rx')
    graph[1].plot(initial_iteration_centroids[2:, 0], initial_iteration_centroids[2:, 1], 'cx')


    for b in range(9):
        distanceBetPointsAndCentroids(Y)
        final_iteration_centroids = return_best(Y)
        print("at iteration ", b + 1, "centroid values are: ", final_iteration_centroids)

    min_values = return_closest(Y)

    for a in range(len(min_values)):
        if min_values[a] == 0:
            graph[2].scatter(x[a],y[a], c = 'y')
        elif min_values[a] == 1:
            graph[2].scatter(x[a],y[a], c = 'r')
        elif min_values[a] == 2:
            graph[2].scatter(x[a],y[a], c = 'c')
    
    graph[2].plot(final_iteration_centroids[0:, 0], final_iteration_centroids[0:, 1], 'yx')
    graph[2].plot(final_iteration_centroids[1:, 0], final_iteration_centroids[1:, 1], 'rx')
    graph[2].plot(final_iteration_centroids[2:, 0], final_iteration_centroids[2:, 1], 'cx')

    plt.show()

    
datafile = 'kmeansdata.mat'
data = scipy.io.loadmat(datafile)


X = np.array([[1,1],[2,1],[4,3],[5,4]]) # class activity

X = data['X']
print("len of x is: ", len(X))

num_points = len(X)

K = 3 # class activity

initial_centroids = np.array([[3.0,3.0], [6.0,2.0], [8.0,5.0]])
print("initial_centroids is : ", initial_centroids)

dist = np.zeros((K,num_points)) # rows by columns / centroid by data

Y = np.zeros((K,num_points)) # 3 by 300

logic(data,Y)
