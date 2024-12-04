#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:37:44 2023

@author: gmartinezarellano
"""

import matplotlib as plt
import pickle

machine = "M02"
operation = "OP08"
dateToUse = "ALLDATA"
test_size = 0.3
path_of_directory= '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation

train_x = pickle.load(open(path_of_directory+'/'+dateToUse+'_train_x_'+str(1-test_size)+'.pkl', 'rb'))
train_y = pickle.load(open(path_of_directory+'/'+dateToUse+'_train_y_'+str(1-test_size)+'.pkl', 'rb'))
test_x = pickle.load(open(path_of_directory+'/'+dateToUse+'_test_x_'+str(test_size)+'.pkl', 'rb'))
test_y = pickle.load(open(path_of_directory+'/'+dateToUse+'_test_y_'+str(test_size)+'.pkl', 'rb'))
#signal = train_x[:1,:1,:]
print(test_y)

#synthetic_x_train = pickle.load(open(path_of_directory+'/'+dateToUse+"_softdtw_synthetic_x_train_300.pkl", 'rb'))

plt.pyplot.plot(test_x[22,0,:])
#plt.pyplot.plot(synthetic_x_train[42,0,:])