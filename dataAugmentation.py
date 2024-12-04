#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:52:41 2023

@author: gmartinezarellano
"""

import numpy as np



def ts_df_to_array(ts_df, index):
    return np.array([ts_df.iloc[index].iloc[i].values for i in range(len(ts_df.iloc[0]))])

def ts_df_to_arrays(ts_df, swapaxes=False):
    arrays=[]
    for index in range(len(ts_df)):
        array = ts_df_to_array(ts_df, index)
        if swapaxes:
            arrays.append(np.swapaxes(array,0,1))
        else:
            arrays.append(array)
    return np.array(arrays)

def softdtw_augment_train_set(x_train, y_train, classes, num_synthetic_ts, max_neighbors=5): 
    from tslearn.neighbors import KNeighborsTimeSeries
    from tslearn.barycenters import softdtw_barycenter
    from tslearn.metrics import gamma_soft_dtw
    
    # synthetic train set and labels 
    synthetic_x_train = []
    synthetic_y_train = []

    # loop through each class
    for c in classes:
        # get the MTS for this class 
        c_x_train = x_train[np.where(y_train==c)]
        print(len(c_x_train))
        if len(c_x_train) == 1 :
            # skip if there is only one time series per set
            continue
        # compute appropriate gamma for softdtw for the entire class
        class_gamma = gamma_soft_dtw(c_x_train)
        # loop through the number of synthtectic examples needed
        generated_samples = 0
        while generated_samples < num_synthetic_ts:
            # Choose a random representative for the class
            representative_indices = np.arange(len(c_x_train))
            random_representative_index = np.random.choice(representative_indices, size=1, replace=False)
            random_representative = c_x_train[random_representative_index]
            # Choose a random number of neighbors (between 1 and one minus the total number of class representatives)
            random_number_of_neighbors = int(np.random.uniform(1, max_neighbors, size=1))
            knn = KNeighborsTimeSeries(n_neighbors=random_number_of_neighbors+1, metric='softdtw', metric_params={'gamma': class_gamma}).fit(c_x_train)
            random_neighbor_distances, random_neighbor_indices = knn.kneighbors(X=random_representative, return_distance=True)
            random_neighbor_indices = random_neighbor_indices[0]
            random_neighbor_distances = random_neighbor_distances[0]
            nearest_neighbor_distance = np.sort(random_neighbor_distances)[1]
            random_neighbors = np.zeros((random_number_of_neighbors+1, c_x_train.shape[1], c_x_train.shape[2]), dtype=float)
            for j, neighbor_index in enumerate(random_neighbor_indices):
                random_neighbors[j,:] = c_x_train[neighbor_index]
            # Choose a random weight vector (and then normalize it)
            weights = np.exp(np.log(0.5)*random_neighbor_distances/nearest_neighbor_distance)
            weights /= np.sum(weights)
            # Compute tslearn.barycenters.softdtw_barycenter with weights=random weights and gamma value specific to neighbors
            random_neighbors_gamma = gamma_soft_dtw(random_neighbors)
            generated_sample = softdtw_barycenter(random_neighbors, weights=weights, gamma=random_neighbors_gamma)
            synthetic_x_train.append(generated_sample)
            synthetic_y_train.append(c)         
            # Repeat until you have the desired number of synthetic samples for each class
            generated_samples += 1
            print(generated_samples)
    # return the synthetic set 
    
    return np.array(synthetic_x_train), np.array(synthetic_y_train)

