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

    def __init__(self, num_classes, input_size,config):
        super(tsCNNConfigurable, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_size, config['filter1'], kernel_size=config['kernel1'], stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(config['filter1'], config['filter2'], kernel_size=config['kernel2'], padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv1d(config['filter2'], config['filter3'], kernel_size=config['kernel3'], padding=1),
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
        self.classifier = nn.Sequential(nn.Linear(config['filter3'], num_classes))


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x