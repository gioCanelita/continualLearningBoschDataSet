#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:25:48 2023

@author: gmartinezarellano
"""

from  DataLoading import generateTrainAndTestFiles, generateALLLDATATestFiles


dateToUse = "ALLDATA"
operation = "OP07"
machine = "M02"

#generateTrainAndTestFiles(dateToUse, operation, machine, 0.3, True)
generateALLLDATATestFiles(dateToUse,operation,machine)

