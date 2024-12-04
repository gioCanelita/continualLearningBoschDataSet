#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:18:09 2023

@author: gmartinezarellano
"""



from avalanche.models import IncrementalClassifier
import torch.nn as nn

class tsCNNConfigurable(nn.Module):
    """
    Convolutional Neural Network

    **Example**::

        >>> from avalanche.models import SimpleCNN
        >>> n_classes = 10 # e.g. MNIST
        >>> model = SimpleCNN(num_classes=n_classes)
        >>> print(model) # View model details
    """

    def __init__(self, num_classes, input_size, filter1,kernel1,filter2,kernel2,filter3,kernel3):
        super(tsCNNConfigurable, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_size, filter1, kernel_size=kernel1, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(filter1, filter2, kernel_size=kernel2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv1d(filter2, filter3, kernel_size=kernel3, padding=1),
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
        self.classifier = nn.Sequential(nn.Linear(64, num_classes))


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x