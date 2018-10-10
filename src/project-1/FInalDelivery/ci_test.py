# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:48:27 2018

@author: Lars
"""


import pandas as pd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

## amount of data
data = pd.read_csv("data10.csv")




## Firstly X is independent of all else
#X = data.drop(["y", "A", "foreign_A202"], axis = 1)
A = data.A
Y = data.y
Z = data.foreign_A202


            
## Now measure the distribution of A for each value of Y, for different values of Z
##
n_figures = 0

for y in [1, 2]:
    ## P(A | Y, Z = 1)
    positive = (Y==y) & (Z==1)
    positive_alpha = sum(A[positive]==1)
    positive_beta = sum(A[positive]==0)
    positive_ratio = positive_alpha / (positive_alpha + positive_beta)


    ## P(A | Y, Z = - 1)
    negative = (Y==y) & (Z==0)
    negative_alpha = sum(A[negative]==1)
    negative_beta = sum(A[negative]==0)
    negative_ratio = negative_alpha / (negative_alpha + negative_beta)

    print("y: ", y, "Deviation: ", abs(positive_ratio - negative_ratio))

"""
y:  1 Deviation:  0.0981424148606811
y:  2 Deviation:  0.03571428571428571
"""