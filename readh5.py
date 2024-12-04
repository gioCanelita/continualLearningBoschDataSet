# -*- coding: utf-8 -*-
"""
Spyder Editor

Giovanna Martinez Arellano
16/12/22

Loads one OPerations good and bad cases

good is 0

bad is 1

TODO augment bad class to create balanced classes in loaded data set
"""



#import torch
#from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import pickle
import numpy as np
import torch
import os
from  datetime import datetime
from torch.utils.tensorboard import SummaryWriter
#from imblearn.over_sampling import SMOTE

#from sklearn.model_selection import train_test_split
from avalanche.models import SimpleMLP
import avalanche as avl
from avalanche.training.supervised import Naive, EWC, SynapticIntelligence
#from avalanche.benchmarks.scenarios import CLExperience, CLScenario
from torch.optim import SGD
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.benchmarks.generators import ni_benchmark, tensors_benchmark


from DataLoading import getOrderedDataFiles, augmentClass
from DataLoading import getSamplesFromFiles, getTrainAndTestTensors
from DataLoading import MyDataset, loadDataForTesting, multipleMachinesIncrementalOperations, loadDataForExperiment, multipleOperationsIncrementalTime, loadDataForYearTesting, loadDataMachineTesting
from dataAugmentation import softdtw_augment_train_set
from contLearningModel import tsCNN, train_one_epoch

from torch.utils.data import Dataset, TensorDataset, DataLoader

from avalanche.training.plugins.checkpoint import CheckpointPlugin, \
    FileSystemCheckpointStorage

def continualLearning(checkpointpath, dateToUse,operation,machine, epochs, iterationNumber):
    train_x, test_x, train_y, test_y = loadDataForExperiment(dateToUse, operation, machine,0.3)
    
    train, test = getTrainAndTestTensors(train_x, test_x, train_y, test_y)
    
    model = tsCNN(2,3)
    if os.path.exists(checkpointpath+str(iterationNumber)+'.pt'):
        checkpoint = torch.load(checkpointpath+str(iterationNumber)+'.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        optimiser = SGD(model.parameters(), lr=0.001, momentum=0.9)
        #epoch = checkpoint['epoch']
        #loss = checkpoint['loss']
        bench = ni_benchmark(train_dataset=train, test_dataset=test, n_experiences=1)
           
        # log to Tensorboard
        tb_logger = TensorboardLogger()
        logname = 'log_'+str(iterationNumber)+'.txt'
        # log to text file
        text_logger = TextLogger(open(logname, 'a'))

        # print to stdout
        interactive_logger = InteractiveLogger()   

        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            timing_metrics(epoch=True, epoch_running=True),
            forgetting_metrics(experience=True, stream=True),
            cpu_usage_metrics(experience=True),
            confusion_matrix_metrics(num_classes=2, save_image=False,
                                     stream=True),
            disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loggers=[interactive_logger, text_logger, tb_logger]
        )
        
        cl_strategy = EWC(
            model, optimiser,
            CrossEntropyLoss(), train_mb_size=10, train_epochs=epochs, eval_mb_size=100,
            evaluator=eval_plugin)
        
        #cl_strategy = AR1(criterion=CrossEntropyLoss(), device="cpu")
 
        
        train_stream = bench.train_stream
        results = []
        for experience in train_stream:
            cl_strategy.train(experience)
            #experience.dataset
            print('Training completed')
            results.append(cl_strategy.eval(bench.test_stream))
            print(results)
            torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'loss': cl_strategy.loss,
                },checkpointpath+str(iterationNumber+1)+'.pt')
        
    return train_x, train_y, test_x, test_y
def runEvaluation(checkpointpath,testDataPath):
    
    test_x, test_y = loadDataForTesting(dateToUse, operation, machine)
    test_x = test_x.astype('float32')
    test_y = test_y.astype('int64')
    print(test_x)
    print(test_y)
    test= TensorDataset(torch.tensor(test_x),torch.from_numpy(test_y))
    print(test)    
    epochs=50
    model = tsCNN(2,3)
    if os.path.exists(checkpointpath+'.pt'):
        checkpoint = torch.load(checkpointpath+'.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        optimiser = SGD(model.parameters(), lr=0.001, momentum=0.9)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        bench = ni_benchmark(train_dataset=test, test_dataset=test, n_experiences=1)
        model.eval()
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            timing_metrics(epoch=True, epoch_running=True),
            forgetting_metrics(experience=True, stream=True),
            cpu_usage_metrics(experience=True),
            confusion_matrix_metrics(num_classes=2, save_image=False,
                                     stream=True),
            disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True)
            #loggers=[interactive_logger, text_logger, tb_logger]
        )
        
        cl_strategy = EWC(
            model, optimiser, CrossEntropyLoss(),ewc_lambda=98.5,
            mode="separate", train_mb_size=10, train_epochs=epochs, eval_mb_size=100,
            evaluator=eval_plugin)
        
        print(cl_strategy.eval(bench.test_stream))
        
def runEvaluationPerYear(checkpointpath, dateToUSe, operation, machines): #for one machiune (train 3 operations and test another one) but year incremental
    
    test_x, test_y = loadDataForYearTesting(dateToUse, operation, machines)
    test_x = test_x.astype('float32')
    test_y = test_y.astype('int64')
    print(test_x)
    print(test_y)
    test= TensorDataset(torch.tensor(test_x),torch.from_numpy(test_y))
    print(test)    
    epochs=50
    model = tsCNN(2,3)
    if os.path.exists(checkpointpath+'.pt'):
        checkpoint = torch.load(checkpointpath+'.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        optimiser = SGD(model.parameters(), lr=0.001, momentum=0.9)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        bench = ni_benchmark(train_dataset=test, test_dataset=test, n_experiences=1)
        model.eval()
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            timing_metrics(epoch=True, epoch_running=True),
            forgetting_metrics(experience=True, stream=True),
            cpu_usage_metrics(experience=True),
            confusion_matrix_metrics(num_classes=2, save_image=False,
                                     stream=True),
            disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True)
            #loggers=[interactive_logger, text_logger, tb_logger]
        )
        
        cl_strategy = EWC(
            model, optimiser, CrossEntropyLoss(),ewc_lambda=98.5,
            mode="separate", train_mb_size=10, train_epochs=epochs, eval_mb_size=100,
            evaluator=eval_plugin)
        
        print(cl_strategy.eval(bench.test_stream))
        
def runEvaluationOneMachine(checkpointpath,operation, machine): #for one machiune (train 3 operations and test another one) but year incremental
    
    test_x, test_y = loadDataMachineTesting(operation, machine)
    test_x = test_x.astype('float32')
    test_y = test_y.astype('int64')
    print(test_x)
    print(test_y)
    test= TensorDataset(torch.tensor(test_x),torch.from_numpy(test_y))
    print(test)    
    epochs=50
    model = tsCNN(2,3)
    if os.path.exists(checkpointpath+'.pt'):
        checkpoint = torch.load(checkpointpath+'.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        optimiser = SGD(model.parameters(), lr=0.001, momentum=0.9)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        bench = ni_benchmark(train_dataset=test, test_dataset=test, n_experiences=1)
        model.eval()
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            timing_metrics(epoch=True, epoch_running=True),
            forgetting_metrics(experience=True, stream=True),
            cpu_usage_metrics(experience=True),
            confusion_matrix_metrics(num_classes=2, save_image=False,
                                     stream=True),
            disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True)
            #loggers=[interactive_logger, text_logger, tb_logger]
        )
        
        cl_strategy = EWC(
            model, optimiser, CrossEntropyLoss(),ewc_lambda=98.5,
            mode="separate", train_mb_size=10, train_epochs=epochs, eval_mb_size=100,
            evaluator=eval_plugin)
        
        print(cl_strategy.eval(bench.test_stream))
        
#This will train the model for one machine with several experiences but without any CL strategy
def runTrainingWithoutCL(iterationNumber, dateToUse, machine, operation,epochs_p, testMachine):
    
    experience_train = []
    train_x, test_x, train_y, test_y = loadDataForExperiment(dateToUse, operation,machine,0.3, testMachine)
    
    ######################################  DATA AUGMENTATION ############################
    
    num_synthetic_ts = np.count_nonzero(train_y == 0)
    path_augmented = '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation+'/'
    
    
    synthetic_x_train, synthetic_y_train = softdtw_augment_train_set(train_x, train_y, [1], num_synthetic_ts)

    
    train_x = np.concatenate((train_x,synthetic_x_train))
    train_x = train_x.astype('float32')

    train_y = np.concatenate((train_y, synthetic_y_train))
    train_y = train_y.astype('int64')
    train, test = getTrainAndTestTensors(train_x, test_x, train_y, test_y)

    
    
    ########################## LOADING DATA FOR THE FOLLOWING EXPERIENCES#########
    
    
    train_x2, test_x2, train_y2, test_y2 = loadDataForExperiment("ALLDATA", "OP11", machine,0.3, testMachine)
    
    num_synthetic_ts = np.count_nonzero(train_y2 == 0)
    path_augmented = '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation+'/'
    #
    
    synthetic_x_train, synthetic_y_train = softdtw_augment_train_set(train_x2, train_y2, [1], num_synthetic_ts)
    train_x2 = np.concatenate((train_x2,synthetic_x_train))
    train_x2 = train_x2.astype('float32')
    train_y2 = np.concatenate((train_y2, synthetic_y_train))
    train_y2 = train_y2.astype('int64')

    
    
    train_x3, test_x3, train_y3, test_y3 = loadDataForExperiment("ALLDATA", "OP12", machine,0.3, testMachine)
    
    num_synthetic_ts = np.count_nonzero(train_y3 == 0)
    path_augmented = '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation+'/'
    #
    
    synthetic_x_train, synthetic_y_train = softdtw_augment_train_set(train_x3, train_y3, [1], num_synthetic_ts)
    train_x3 = np.concatenate((train_x3,synthetic_x_train))
    train_x3 = train_x3.astype('float32')
    train_y3 = np.concatenate((train_y3, synthetic_y_train))
    train_y3 = train_y3.astype('int64')

    
    experience_train.append((torch.tensor(train_x), torch.from_numpy(train_y)))
    experience_train.append((torch.tensor(train_x2), torch.from_numpy(train_y2)))
    experience_train.append((torch.tensor(train_x3), torch.from_numpy(train_y3)))
    
################################### Creating model ########################################
    model = tsCNN(2,3)
    print(model)
    optimiser = SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    #model = avl.models.SimpleSequenceClassifier(input_size = 2000*3,hidden_size=3000,n_classes=2,rnn_layers=1)
    #optimiser = Adam(model.parameters(), lr=0.001)
    

    
        # log to Tensorboard
    tb_logger = TensorboardLogger()
    logname = 'log_Exp_1_17_Feb_24_M02_3Operations.txt'
    # log to text file
    text_logger = TextLogger(open(logname, 'a'))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=2, save_image=False,
                                 stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # CREATE THE STRATEGY INSTANCE (NAIVE) if no checkpoint
    #cl_strategy = EWC(
    #    model, optimiser, CrossEntropyLoss(),ewc_lambda=102.0,
    #    mode="separate", train_mb_size=10, train_epochs=epochs, eval_mb_size=100,
    #    evaluator=eval_plugin)



    #bench = ni_benchmark(train_dataset=[train,trainEXP2,trainEXP3], test_dataset=[test,testEXP2,testEXP3], n_experiences=3)
    #bench = tensors_benchmark(
    #train_tensors=experience_train,
    #test_tensors=experience_test,
    #task_labels=[0, 0,0,0],  # Task label of each train exp
    #complete_test_set_only=False
    #)
    
    #train_stream = bench.train_stream
    results = []
        # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(type(experience_train))
    print(len(experience_train))
    i=0
    for experience in experience_train:
        print(experience)
        
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/NOCL_CNN_trainer_2_1_APRIL_{}'.format(timestamp))
        epoch_number = 0

        EPOCHS = epochs_p

        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

        #    Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss = train_one_epoch(epoch_number, writer, model, optimiser, loss_fn, experience)

            # We don't need gradients on to do reporting
            model.train(False)

            running_vloss = 0.0
            #for i, vdata in enumerate(validation_loader):
            #    vinputs, vlabels = vdata
            #    voutputs = model(vinputs)
            #    vloss = loss_fn(voutputs, vlabels)
             #   running_vloss += vloss

            avg_vloss = running_vloss / (0 + 1)
            print('LOSS train {}'.format(avg_loss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training Loss',
                    { 'Training' : avg_loss},
                    epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            #if avg_vloss < best_vloss:
            #    best_vloss = avg_vloss
            #    model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                

            epoch_number += 1
        i = i+1
        
        torch.save({
            'epoch': epoch_number,
            'model_state_dict': model.state_dict(),
            'loss': avg_loss,
            },checkpointpath+str(i)+'.pt')
        print(i)
    
    return 0
def runTrainingIncrementalOperationsWithoutCL(iterationNumber, machines, operations, epochs_p, machineTest):
    

    experience_train, experience_test = multipleMachinesIncrementalOperations(machines,operations,machineTest)
    model_path = './checkpoints/Exp_2_1_April_23_2Machines_MultipleOps/singleTask_NOCL_incremental1_'
    
    #model = SimpleMLP(num_classes=2, input_size = 2000*3, hidden_layers=2)
    model = tsCNN(2,3)
    print(model)
    optimiser = SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    #model = avl.models.SimpleSequenceClassifier(input_size = 2000*3,hidden_size=3000,n_classes=2,rnn_layers=1)
    #optimiser = Adam(model.parameters(), lr=0.001)
    

    
        # log to Tensorboard
    tb_logger = TensorboardLogger()
    logname = 'log_Exp_2_1_April_23_2Machines_4_Operations.txt'
    # log to text file
    text_logger = TextLogger(open(logname, 'a'))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=2, save_image=False,
                                 stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # CREATE THE STRATEGY INSTANCE (NAIVE) if no checkpoint
    #cl_strategy = EWC(
    #    model, optimiser, CrossEntropyLoss(),ewc_lambda=102.0,
    #    mode="separate", train_mb_size=10, train_epochs=epochs, eval_mb_size=100,
    #    evaluator=eval_plugin)



    #bench = ni_benchmark(train_dataset=[train,trainEXP2,trainEXP3], test_dataset=[test,testEXP2,testEXP3], n_experiences=3)
    #bench = tensors_benchmark(
    #train_tensors=experience_train,
    #test_tensors=experience_test,
    #task_labels=[0, 0,0,0],  # Task label of each train exp
    #complete_test_set_only=False
    #)
    
    #train_stream = bench.train_stream
    results = []
        # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(type(experience_train))
    print(len(experience_train))
    i=0
    for experience in experience_train:
        print(experience)
        
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/NOCL_CNN_trainer_2_1_APRIL_{}'.format(timestamp))
        epoch_number = 0

        EPOCHS = epochs_p

        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

        #    Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss = train_one_epoch(epoch_number, writer, model, optimiser, loss_fn, experience)

            # We don't need gradients on to do reporting
            model.train(False)

            running_vloss = 0.0
            #for i, vdata in enumerate(validation_loader):
            #    vinputs, vlabels = vdata
            #    voutputs = model(vinputs)
            #    vloss = loss_fn(voutputs, vlabels)
             #   running_vloss += vloss

            avg_vloss = running_vloss / (0 + 1)
            print('LOSS train {}'.format(avg_loss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training Loss',
                    { 'Training' : avg_loss},
                    epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            #if avg_vloss < best_vloss:
            #    best_vloss = avg_vloss
            #    model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                

            epoch_number += 1
        i = i+1
        
        torch.save({
            'epoch': epoch_number,
            'model_state_dict': model.state_dict(),
            'loss': avg_loss,
            },model_path+str(i)+'.pt')
        print(i)
    
    return 0  
def runTrainingIncrementalOperations(iterationNumber, machines, operations, epochs, machineTest):
    

    experience_train, experience_test = multipleMachinesIncrementalOperations(machines,operations,machineTest)
    #experience_train, experience_test = multipleOperationsIncrementalTime(['OP01','OP02', 'OP07'],['2019','2020', '2021'], 'OP04')
    
    #model = SimpleMLP(num_classes=2, input_size = 2000*3, hidden_layers=2)
    model = tsCNN(2,3)
    print(model)
    optimiser = SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    #model = avl.models.SimpleSequenceClassifier(input_size = 2000*3,hidden_size=3000,n_classes=2,rnn_layers=1)
    #optimiser = Adam(model.parameters(), lr=0.001)
    

    
        # log to Tensorboard
    tb_logger = TensorboardLogger()
    logname = 'log_Exp_1_12_Feb_2Machines_4_Operations_L102.txt'
    # log to text file
    text_logger = TextLogger(open(logname, 'a'))

    # print to stdou2
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=2, save_image=False,
                                 stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )


    # CREATE THE STRATEGY INSTANCE (NAIVE) if no checkpoint
    cl_strategy = EWC(
        model, optimiser, CrossEntropyLoss(),ewc_lambda=102.0,
        mode="separate", train_mb_size=10, train_epochs=epochs, eval_mb_size=100,
        evaluator=eval_plugin)


    #Second CL strategy, commenting the previous one
    #cl_strategy = SynapticIntelligence(model, optimiser, CrossEntropyLoss(),si_lambda=70, train_mb_size=10, train_epochs=epochs, eval_mb_size=100,evaluator=eval_plugin)

    #bench = ni_benchmark(train_dataset=[train,trainEXP2,trainEXP3], test_dataset=[test,testEXP2,testEXP3], n_experiences=3)
    bench = tensors_benchmark(
    train_tensors=experience_train,
    test_tensors=experience_test,
    task_labels=[0, 0,0,0],  # Task label of each train exp
    complete_test_set_only=False
    )
    
    train_stream = bench.train_stream
    results = []


    for experience in train_stream:
        cl_strategy.train(experience)
        #experience.dataset
        print('Training completed')
        results.append(cl_strategy.eval(bench.test_stream))
        print(results)
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'loss': cl_strategy.loss,
            },checkpointpath+str(iterationNumber)+'.pt')
        iterationNumber = iterationNumber+1
    
    return 0
        
    
def runTraining(iterationNumber, dateToUse, machine, operation,epochs, testMachine):
    
    

    
    #epochs=50
    #samples_x, samples_y = loadDataForExperiment(dateToUse, operation, machine)
    
    
    #dataSet = MyDataset(np.asarray(samples_x),np.asarray(samples_y),10)
    #train_x,test_x, train_y, test_y = dataSet.getTrainAndTestArrays()
    
    train_x, test_x, train_y, test_y = loadDataForExperiment(dateToUse, operation,machine,0.3, testMachine)
    
    ######################################  DATA AUGMENTATION ############################
    
    num_synthetic_ts = np.count_nonzero(train_y == 0)
    path_augmented = '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation+'/'
    #
    
    synthetic_x_train, synthetic_y_train = softdtw_augment_train_set(train_x, train_y, [1], num_synthetic_ts)

    
    train_x = np.concatenate((train_x,synthetic_x_train))
    train_x = train_x.astype('float32')
    train_y = np.concatenate((train_y, synthetic_y_train))
    
    train, test = getTrainAndTestTensors(train_x, test_x, train_y, test_y)

    
    
    ########################## LOADING DATA FOR THE FOLLOWING EXPERIENCES#########
    
    
    train_x2, test_x2, train_y2, test_y2 = loadDataForExperiment("ALLDATA", "OP11", machine,0.3, testMachine)
    
    num_synthetic_ts = np.count_nonzero(train_y2 == 0)
    path_augmented = '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation+'/'
    #
    
    synthetic_x_train, synthetic_y_train = softdtw_augment_train_set(train_x2, train_y2, [1], num_synthetic_ts)
    train_x2 = np.concatenate((train_x2,synthetic_x_train))
    train_x2 = train_x2.astype('float32')
    train_y2 = np.concatenate((train_y2, synthetic_y_train))
    
    trainEXP2, testEXP2 = getTrainAndTestTensors(train_x2, test_x2, train_y2, test_y2)
    
    
    train_x3, test_x3, train_y3, test_y3 = loadDataForExperiment("ALLDATA", "OP12", machine,0.3, testMachine)
    
    num_synthetic_ts = np.count_nonzero(train_y3 == 0)
    path_augmented = '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation+'/'
    #
    
    synthetic_x_train, synthetic_y_train = softdtw_augment_train_set(train_x3, train_y3, [1], num_synthetic_ts)
    train_x3 = np.concatenate((train_x3,synthetic_x_train))
    train_x3 = train_x3.astype('float32')
    train_y3 = np.concatenate((train_y3, synthetic_y_train))
    
    trainEXP3, testEXP3 = getTrainAndTestTensors(train_x3, test_x3, train_y3, test_y3)
    
    print(train_y.size)
    print(train_y2.size)
    print(train_y3.size)
    
    print(train)
    print(trainEXP2)
    print(trainEXP3)
    ########################## MODEL DEFINITION ########################################
    
    #model = SimpleMLP(num_classes=2, input_size = 2000*3, hidden_layers=2)
    model = tsCNN(2,3)
    print(model)
    optimiser = SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    #model = avl.models.SimpleSequenceClassifier(input_size = 2000*3,hidden_size=3000,n_classes=2,rnn_layers=1)
    #optimiser = Adam(model.parameters(), lr=0.001)
    

    
        # log to Tensorboard
    tb_logger = TensorboardLogger()
    logname = 'log_exp2_18_Feb_24_M02.txt'
    # log to text file
    text_logger = TextLogger(open(logname, 'a'))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=2, save_image=False,
                                 stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )


    # CREATE THE STRATEGY INSTANCE (NAIVE) if no checkpoint
    #cl_strategy = EWC(
    #    model, optimiser, CrossEntropyLoss(),ewc_lambda=102.0,
    #    mode="separate", train_mb_size=10, train_epochs=epochs, eval_mb_size=100,
    #    evaluator=eval_plugin)

    cl_strategy = SynapticIntelligence(model, optimiser, CrossEntropyLoss(),si_lambda=100, train_mb_size=10, train_epochs=epochs, eval_mb_size=100,evaluator=eval_plugin)


    #bench = ni_benchmark(train_dataset=[train,trainEXP2,trainEXP3], test_dataset=[test,testEXP2,testEXP3], n_experiences=3)
    bench = tensors_benchmark(
    train_tensors=[(torch.tensor(train_x), torch.from_numpy(train_y)), (torch.tensor(train_x2), torch.from_numpy(train_y2)),(torch.tensor(train_x3), torch.from_numpy(train_y3))],
    test_tensors=[(torch.tensor(test_x), torch.from_numpy(test_y)),(torch.tensor(test_x2),torch.from_numpy(test_y2)),(torch.tensor(test_x3),torch.from_numpy(test_y3))],
    task_labels=[0, 0,0],  # Task label of each train exp
    complete_test_set_only=False
    )
    
    train_stream = bench.train_stream
    results = []
    for experience in train_stream:
        cl_strategy.train(experience)
        #experience.dataset
        print('Training completed')
        results.append(cl_strategy.eval(bench.test_stream))
        print(results)
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'loss': cl_strategy.loss,
            },checkpointpath+str(iterationNumber)+'.pt')
        iterationNumber = iterationNumber+1
    
    return train_x, train_y, test_x, test_y
        



dateToUse = "ALLDATA" 

machine = "M02"
testMachine = "M02"
operation = "OP12"

iterationNumber =3
trainDataPath = '../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation+'/'

checkpointpath = './checkpoints/Exp_2_18_Feb_24_M02_MultipleOps/singleTask_incremental_'
#train_x, train_y, test_x, test_y = runTraining(iterationNumber, dateToUse, machine, operation,20, testMachine)

runEvaluationOneMachine(checkpointpath+str(iterationNumber), operation, testMachine)

#runEvaluation(checkpointpath+str(iterationNumber),'.../Data/boschresearch CNC_Machining main data/'+machine+'/'+operation+'/'+dateToUse+'/')
#runEvaluationPerYear(checkpointpath+str(iterationNumber),dateToUse, operation, machines)
#train_x, train_y, test_x, test_y = continualLearning(checkpointpath,dateToUse,operation,machine,50,iterationNumber)


#######################################################
#machines = ['M01','M02']
#operations = ['OP08','OP10','OP11','OP12']


#runTrainingIncrementalOperations(iterationNumber, machines, operations, 20, 'M03')



#runTrainingIncrementalOperationsWithoutCL(iterationNumber, machines, operations, 20, testMachine)


#runTrainingWithoutCL(iterationNumber, dateToUse, machine, operation,20, testMachine)
#    





