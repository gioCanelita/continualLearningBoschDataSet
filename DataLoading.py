#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:16:39 2023

@author: gmartinezarellano
"""

import torch
from torch.utils.data import Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import h5py
from matplotlib import pyplot
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from einops.layers.torch import Rearrange

from dataAugmentation import softdtw_augment_train_set

class MyDataset(Dataset):
    def __init__(self, Samples,SamplesLabels, q):
        
        #train, test = train_test_split(dfSamples,dfSamplesLabels, test_size=0.2, shuffle=True)
        self.data = Samples
        self.q = q
        #self.train = train
        #self.test = test
        self.Samples = Samples
        self.SamplesLabels = SamplesLabels

    def __len__(self):
        return self.data.shape[0] // self.q

    def __getitem__(self, index):
        return self.data[index * self.q: (index+1) * self.q]
    
    def getTrainAndTestArrays(self, test_size):
        
        train_x, test_x, train_y, test_y = train_test_split(self.Samples,self.SamplesLabels, test_size = test_size, shuffle=True)
        train_x = train_x.astype('float32')
        test_x = test_x.astype('float32')
        
        return train_x, test_x, train_y, test_y
    
        
    def normalise(self, x):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        self.max = x.max()
        self.min = x.min()
        return (2 * (x - x.min())/(x.max() - x.min()) - 1)
    
    #def train_test_split(samples,sampleLables,trainPercentage):
#        train_y=[]
#    #    train_x = []
#        test_x=[]
#        test_y=[]
#        trainSize = len(samples*trainPercentage)
        
 #       for i in trainSize:
 #           print(i)
            
 #       return train_x, train_y,  test_x, test_y
    
def augmentClass( train_x, train_y, classToAugment,  num_synthetic_ts, path, nameFile):
    
    #augmented_x = []
    #augmented_y = []
    classes = [classToAugment]
    
    synthetic_x_train, synthetic_y_train = softdtw_augment_train_set(train_x, train_y, classes, num_synthetic_ts)
    
    #augmented_x = train_x.append(synthetic_x_train)

    #augmented_x = np.concatenate((train_x,synthetic_x_train))
    #augmented_y = np.concatenate((train_y, synthetic_y_train))
    
    #augmented_x = augmented_x.astype('float32')

    #synthetic_x_train = synthetic_x_train.transpose(0,2,1)
    #synthetic_x_train = Rearrange('b w c -> b c w')(synthetic_x_train)
    
    #augmented_x = train_x.extend(synthetic_x_train)
    #augmented_y = train_y.extend(synthetic_y_train)
    pickle.dump(synthetic_x_train, open("%s%s_softdtw_synthetic_x_train_%d.pkl" % (path,nameFile, num_synthetic_ts), 'wb'))
    pickle.dump(synthetic_y_train, open("%s%s_softdtw_synthetic_y_train_%d.pkl" % (path,nameFile, num_synthetic_ts), 'wb'))

    
    return 0

###########################################################################

def getOrderedDataFiles(path, date, allData=True):
    Twenty19 = []
    Twenty20 = []
    Twenty21 = []
    temp = []
    orderedDataFiles = []
    print(date)
    for filename in sorted(os.listdir(path)):
        f = os.path.join(path,filename)
        if (allData):
            if ("2019" in filename):
                Twenty19.append(f)
            elif ("2020" in filename):
                Twenty20.append(f)
            else:
                Twenty21.append(f)
        else:
            
            if(date in filename and "2019" in date):
                Twenty19.append(f)
            elif (date in filename and "2020" in date):
                Twenty20.append(f)
            elif(date in filename and "2021" in date):
                Twenty21.append(f)
   # for i in Twenty19:
        #print(i)
           
    ############### ordering 2019 data files  ##################
        
    for f in Twenty19:
        if ("Feb" in f):
            temp.append(f)
    
    
    for f in temp:
        if ("Feb" in f):
            #print( "removing: " + f)
            Twenty19.remove(f)
    
    orderedDataFiles.extend(temp)
    orderedDataFiles.extend(Twenty19)
    
    
    ############### ordering 2020 data files  ##################
    
    temp = []
    for f in Twenty20:
        if ("Feb" in f):
            temp.append(f)
    
    
    for f in temp:
        if ("Feb" in f):
            #print( "removing2020: " + f)
            Twenty20.remove(f)
    
    orderedDataFiles.extend(temp)
    orderedDataFiles.extend(Twenty20)
    
    ############### ordering 2021 data files  ##################
    
    temp = []
    for f in Twenty21:
        if ("Feb" in f):
            temp.append(f)
    
    
    for f in temp:
        if ("Feb" in f):
            #print( "removing: 2021 " + f)
            Twenty21.remove(f)
    
    orderedDataFiles.extend(temp)
    orderedDataFiles.extend(Twenty21)
    
    return orderedDataFiles


        
def getSamplesFromFiles(orderedFiles, chunkSize, timeSeriesClass):
        
    samples = []
    samplesLabels=[]
    for file in orderedFiles:
        if os.path.isfile(file):
            data = h5py.File(file,'r')
            data = data['vibration_data']
            data = pd.DataFrame(data)
            #data = minmax_scale(data, feature_range=(-1, 1))
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)


            #data = data.value
            pyplot.plot(data[:,1])

            for i in range(0, len(data), chunkSize):
                if(i+chunkSize > len(data)):
                    end = len(data)
                else:
                    end = i+chunkSize
                #print("chunk: ", i, ",",end)
                #chunk = [timeSeriesClass] 
                #chunk.extend(data[i:end,])
                chunk = data[i:end,]
                #print(chunk.shape)
                samples.append(chunk)
                samplesLabels.append(timeSeriesClass)
        last = samples.pop()
        last = samplesLabels.pop()
        #print(samples)
            
    return samples, samplesLabels
        
def getAllSamplesFromFiles(dateToUse, operation, machine, allData=False):
        
      path_of_the_directory= '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation   
      print(path_of_the_directory)
      orderedFiles = getOrderedDataFiles(path=path_of_the_directory+'/good/' ,date=dateToUse, allData=allData)
      print(orderedFiles)
      goodSamples_x, goodSamples_y = getSamplesFromFiles(orderedFiles,4096, 0) # 0 is good
      
      
      path_of_the_directory= '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation 
      if(os.path.exists(path_of_the_directory+'/bad/')):
          orderedFiles = getOrderedDataFiles(path=path_of_the_directory+'/bad/' , date=dateToUse, allData=allData)
          badSamples_x, badSamples_y = getSamplesFromFiles(orderedFiles,4096, 1)
          goodSamples_x.extend(badSamples_x)
          goodSamples_y.extend(badSamples_y)
          
      print(len(goodSamples_x))
      print(goodSamples_y)
      #
      goodSamples_x = np.asarray(goodSamples_x)
      goodSamples_y = np.asarray(goodSamples_y)
      goodSamples_x = goodSamples_x.transpose(0, 2, 1)
      goodSamples_x = goodSamples_x.astype('float64')
      goodSamples_y = goodSamples_y.astype('int64')
      #dataSet = MyDataset(np.asarray(goodSamples_x),np.a
      #dataSet = MyDataset(np.asarray(goodSamples_x),np.asarray(goodSamples_y),10)
      print("gettingsamples")

      return goodSamples_x, goodSamples_y
    
def generateALLLDATATestFiles(dateToUse, operation,machine):
    
    path_of_the_directory= '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation  
    train_x, train_y = getAllSamplesFromFiles("ALLDATA",operation, machine, allData=True)

    print(train_x)
    pickle.dump(train_x,open(path_of_the_directory+'/'+dateToUse+'_ALLDATA_x_.pkl', 'wb'))
    pickle.dump(train_y,open(path_of_the_directory+'/'+dateToUse+'_ALLDATA_y_.pkl', 'wb'))
    
def generateTrainAndTestFiles(dateToUse, operation, machine,test_size, allData=False):
      #dateToUse = "Aug_2019"

      
      path_of_the_directory= '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation   
      orderedFiles = getOrderedDataFiles(path=path_of_the_directory+'/good/' ,date=dateToUse, allData=allData)
      goodSamples_x, goodSamples_y = getSamplesFromFiles(orderedFiles,4096, 0) # 0 is good
      
      
      path_of_the_directory= '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation 
      if(os.path.exists(path_of_the_directory+'/bad/')):
          orderedFiles = getOrderedDataFiles(path=path_of_the_directory+'/bad/' , date=dateToUse, allData=allData)
          badSamples_x, badSamples_y = getSamplesFromFiles(orderedFiles,4096, 1)
          print(badSamples_y)
          print(len(badSamples_y))
          goodSamples_x.extend(badSamples_x)
          goodSamples_y.extend(badSamples_y)
      
      dataSet = MyDataset(np.asarray(goodSamples_x),np.asarray(goodSamples_y),10)
      train_x,test_x, train_y, test_y = dataSet.getTrainAndTestArrays(test_size)
      
      train_x = train_x.transpose(0, 2, 1)
      #train_x = Rearrange('b w c -> b c w')(train_x)
      #test_x = Rearrange('b w c -> b c w')(test_x)
      test_x = test_x.transpose(0, 2, 1)
      print(train_y)
      print(test_y)
      print(len(train_y))
      print(len(test_y))
      if (allData):
          dateToUse = "ALLDATA"
      pickle.dump(train_x,open(path_of_the_directory+'/'+dateToUse+'_train_x_'+str(1-test_size)+'.pkl', 'wb'))
      pickle.dump(test_x,open(path_of_the_directory+'/'+dateToUse+'_test_x_'+str(test_size)+'.pkl','wb'))
      pickle.dump(train_y,open(path_of_the_directory+'/'+dateToUse+'_train_y_'+str(1-test_size)+'.pkl', 'wb'))
      pickle.dump(test_y,open(path_of_the_directory+'/'+dateToUse+'_test_y_'+str(test_size)+'.pkl', 'wb'))
      
      return 0
  
def loadDataForTesting(dateToUse, operation,machine):
    path_of_directory= '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation

    test_x = pickle.load(open(path_of_directory+'/'+dateToUse+'_ALLDATA_x_.pkl', 'rb'))
    test_y = pickle.load(open(path_of_directory+'/'+dateToUse+'_ALLDATA_y_.pkl', 'rb'))
    
    return test_x, test_y

def loadDataMachineTesting( operation,machine):
    path_of_directory= '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation

    test_x = pickle.load(open(path_of_directory+'/'+'ALLDATA_test_x_0.3.pkl', 'rb'))
    test_y = pickle.load(open(path_of_directory+'/'+'ALLDATA_test_y_0.3.pkl', 'rb'))
    
    return test_x, test_y
    
def loadDataForYearTesting(dateToUse, operation,machines):
    test_y = []
    shape = (0,3, 4096)
    test_x = np.empty(shape)
    for machine in machines:
        test_x_t, test_y_t = getAllSamplesFromFiles(dateToUse,operation, machine, allData=False)
        test_x =  np.concatenate((test_x,test_x_t))
        test_y =  np.concatenate((test_y,test_y_t))
    #test_x = pickle.load(open(path_of_directory+'/'+dateToUse+'_ALLDATA_x_.pkl', 'rb'))
    #test_y = pickle.load(open(path_of_directory+'/'+dateToUse+'_ALLDATA_y_.pkl', 'rb'))
    
    return test_x, test_y
def loadDataForExperiment(dateToUse, operation,machine,test_size, testMachine):
    path_of_directory= '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation
    path_test = '../Data/boschresearch CNC_Machining main data/'+testMachine+'/'+operation
    train_x = pickle.load(open(path_of_directory+'/'+dateToUse+'_train_x_'+str(1-test_size)+'.pkl', 'rb'))
    train_y = pickle.load(open(path_of_directory+'/'+dateToUse+'_train_y_'+str(1-test_size)+'.pkl', 'rb'))
    #test_x = pickle.load(open(path_of_directory+'/'+dateToUse+'_test_x_'+str(test_size)+'.pkl', 'rb'))
    #test_y = pickle.load(open(path_of_directory+'/'+dateToUse+'_test_y_'+str(test_size)+'.pkl', 'rb'))
    test_x = pickle.load(open(path_test+'/'+dateToUse+'_train_x_'+str(1-test_size)+'.pkl', 'rb'))
    test_y = pickle.load(open(path_test+'/'+dateToUse+'_train_y_'+str(1-test_size)+'.pkl', 'rb'))
    
  
    return train_x, test_x, train_y, test_y

#load all operations for a machine and train/test
def loadDataForOneMachineExperiment(dateToUse, operations,machine,test_size, testMachine):
    train_x_accum = []
    train_y_accum = []
    test_x_accum = []
    test_y_accum = []
    
    for operation in operations:
        path_of_directory= '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation
        path_test = '../Data/boschresearch CNC_Machining main data/'+testMachine+'/'+operation
        train_x = pickle.load(open(path_of_directory+'/'+dateToUse+'_train_x_'+str(1-test_size)+'.pkl', 'rb'))
        train_y = pickle.load(open(path_of_directory+'/'+dateToUse+'_train_y_'+str(1-test_size)+'.pkl', 'rb'))
        #test_x = pickle.load(open(path_of_directory+'/'+dateToUse+'_test_x_'+str(test_size)+'.pkl', 'rb'))
        #test_y = pickle.load(open(path_of_directory+'/'+dateToUse+'_test_y_'+str(test_size)+'.pkl', 'rb'))
        test_x = pickle.load(open(path_test+'/'+dateToUse+'_train_x_'+str(1-test_size)+'.pkl', 'rb'))
        test_y = pickle.load(open(path_test+'/'+dateToUse+'_train_y_'+str(1-test_size)+'.pkl', 'rb'))
    
  
    return train_x, test_x, train_y, test_y
    
def getTrainAndTestTensors(train_x, test_x, train_y, test_y):

    #train_x = train_x.astype('float32')
    #test_x = test_x.astype('float32')


    trainTensor = TensorDataset(torch.tensor(train_x),torch.from_numpy(train_y))
    testTensor = TensorDataset(torch.tensor(test_x),torch.from_numpy(test_y))
    return trainTensor, testTensor  
    
def multipleMachinesIncrementalOperations(machines,operations,machineTest):
    
    
    # Here we create training set with two machines and test with the third one, to incrementally learn operations, so each experience has 
    # a anew operation from both machines
    
    experiences_train = []
    experiences_test = []

    #train_x = [[[]]]
    #train_y = [[[]]]
    it=0
    
    for operation in operations:
        shape = (0,3, 4096)
        train_x = np.empty(shape)
        train_y= []
        for machine in machines:
            print(machine)
            train_x_t, train_y_t = getAllSamplesFromFiles("ALLDATA",operation, machine, allData=True)
            train_x = np.concatenate((train_x,train_x_t))
            train_y = np.concatenate((train_y,train_y_t))
           
       # if(it==0): #only generate synthetic data for class NOTOK for the first experience
       #generate synthetic samples for every experience
        count = np.count_nonzero(train_y == 0)
        print(count)
        synthetic_x_train, synthetic_y_train = softdtw_augment_train_set(train_x, train_y, [1], count)
        train_x = np.concatenate((train_x,synthetic_x_train))
        train_y = np.concatenate((train_y,synthetic_y_train))
         #   it = it + 1
        train_x = train_x.astype('float32')
        train_y = train_y.astype('int64')
        experiences_train.append((torch.tensor(train_x), torch.from_numpy(train_y)))
        
        test_x_t, test_y_t = getAllSamplesFromFiles("ALLDATA",operation, machineTest, allData=True)
        test_x_t = test_x_t.astype('float32')
        test_y_t = test_y_t.astype('int64')
        experiences_test.append((torch.tensor(test_x_t), torch.from_numpy(test_y_t)))
        
        print(operation)
        print(len(train_x))
        
        

    return experiences_train, experiences_test


def multipleOperationsIncrementalTime(operations,times, operationTest):
    
    
    #Here we create a training set with the set of operations (all machiines) and test with a different 
    #operation, then incrementally deal with time drift
    experiences_train = []
    experiences_test = []
    machines = ['M01','M02','M03']

    it=0
    
    for time in times:
        shape = (0,3, 4096)
        train_x = np.empty(shape)
        test_x = np.empty(shape)
        train_y= []
        test_y = []
        for operation in operations:
            for machine in machines:
                train_x_t, train_y_t = getAllSamplesFromFiles(time,operation, machine, allData=False)
                print("loading samples from machine", machine+operation+time)
                print(train_x_t.shape)
                train_x = np.concatenate((train_x,train_x_t))
                train_y = np.concatenate((train_y,train_y_t))
                
                        
                test_x_t, test_y_t = getAllSamplesFromFiles(time,operationTest, machine, allData=True)
                test_x = np.concatenate((test_x,test_x_t))
                test_y = np.concatenate((test_y,test_y_t))


        if it==0: #only generate synthetic data for class NOTOK for the first experience
            count = np.count_nonzero(train_y == 0)
            print(count)
            synthetic_x_train, synthetic_y_train = softdtw_augment_train_set(train_x, train_y, [1], count)
            train_x = np.concatenate((train_x,synthetic_x_train))
            train_y = np.concatenate((train_y,synthetic_y_train))
            it = it + 1        
        
        train_x = train_x.astype('float32')
        train_y = train_y.astype('int64')
        experiences_train.append((torch.tensor(train_x), torch.from_numpy(train_y)))
        
        test_x = test_x.astype('float32')
        test_y = test_y.astype('int64')
        experiences_test.append((torch.tensor(test_x), torch.from_numpy(test_y)))
                                
    
    return experiences_train, experiences_test



def oneMachineIncrementalOperations(machine,operations):
    
    
    # Here we create training set with two machines and test with the third one, to incrementally learn operations, so each experience has 
    # a anew operation from both machines
    
    experiences_train = []
    experiences_test = []

    #train_x = [[[]]]
    #train_y = [[[]]]
    it=0
    
    for operation in operations:
        shape = (0,3, 4096)
        train_x = np.empty(shape)
        test_x = np.empty(shape)
        train_y= []
        test_y=[]

        train_x_t, train_y_t = getAllSamplesFromFiles("ALLDATA",operation, machine, allData=True)
        train_x = np.concatenate((train_x,train_x_t))
        train_y = np.concatenate((train_y,train_y_t))
           
        
        #split samples to train and test
        
        
       # if(it==0): #only generate synthetic data for class NOTOK for the first experience
       #generate synthetic samples for every experience
        count = np.count_nonzero(train_y == 0)
        print(count)
        synthetic_x_train, synthetic_y_train = softdtw_augment_train_set(train_x, train_y, [1], count)
        train_x = np.concatenate((train_x,synthetic_x_train))
        train_y = np.concatenate((train_y,synthetic_y_train))
         #   it = it + 1
        train_x = train_x.astype('float32')
        train_y = train_y.astype('int64')
        experiences_train.append((torch.tensor(train_x), torch.from_numpy(train_y)))
        

        test_x = test_x.astype('float32')
        test_y = test_y.astype('int64')
        experiences_test.append((torch.tensor(test_x), torch.from_numpy(test_y)))
        
        print(operation)
        print(len(train_x))
        
        

    return experiences_train, experiences_test




