#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:51:26 2022

@author: gmartinezarellano
"""

from avalanche.benchmarks import SplitMNIST
from avalanche.models import IncrementalClassifier
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_one_epoch(epoch_index, tb_writer, model, optimiser, loss_fn, training_set):
    
    x = training_set[0]
    y = training_set[1]
    print("in pepoch")
    print(len(x))
    print(len(y))
    
    training_loader = DataLoader(TensorDataset(x,y), batch_size=10, shuffle=True)
    #validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        #print(inputs)
        #print(labels)
        # Zero your gradients for every batch!
        optimiser.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimiser.step()

        # Gather data and report
        running_loss += loss.item()
    last_loss = running_loss/len(x)

    return last_loss

class tsCNN(nn.Module):


    def __init__(self, num_classes, input_size):
        super(tsCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=20, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 32, kernel_size=64, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            #nn.Dropout(p=0.25),
            nn.Conv1d(32, 16, kernel_size=64, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv1d(16, 10, kernel_size=3, padding=3),
            #nn.ReLU(inplace=True),
            #nn.MaxPool1d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
            #nn.Conv1d(64, 64, kernel_size=1, padding=0),
            #nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=0.25),
        )
        self.classifier = nn.Sequential(nn.Linear(16, num_classes))


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    
