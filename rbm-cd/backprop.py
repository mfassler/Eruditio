#!/usr/bin/python
# -*- coding: utf-8 -*-


## Copyright 2011, Wizcorp, www.wizcorp.jp

## This code was based heavily upon some matlab code from 
## http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
## which included the following notice: 
#
# Code provided by Ruslan Salakhutdinov and Geoff Hinton
#
# Permission is granted for anyone to copy, use, modify, or distribute this
# program and accompanying programs and documents for any purpose, provided
# this copyright notice is retained and prominently displayed, along with
# a note saying that the original programs are available from our
# web page.
# The programs and documents are distributed without any warranty, express or
# implied.  As the programs were written for research purposes only, they have
# not been tested to the degree that would be advisable in any important
# application.  All use of these programs is entirely at the user's own risk.


import numpy as np
from flattenUtils import *


def backprop_only3(VV, Dim, inputs, targets):

    #### Un-Flatten all of our parameters from the 1-D array
    matrices = multiUnFlatten(VV, Dim)
    W = matrices[0]
    hB = matrices[1]

    # final layer is softmax, not logsig:
    layer3out = np.exp( np.dot(inputs, W) + hB) #numpy auto-tiles hB
    layer3out = layer3out / np.tile( layer3out.sum(1)[:, np.newaxis], (1,10) )

    # We use cross-entropy rather than squared error for our error function:
    # E for error
    ## the ufldl.stanford.edu link calls this the "Cost" function J.
    E = -(targets * np.log(layer3out)).sum(0).sum(0)

    # Classification error:
    # lowercase "delta" from Bishop, page 243, eq: 5.54:  
    #      δ_k = y_k - t_k
    d3 = layer3out - targets

    ## Flatten the gradients into the same shape as VV:
    (df, Dim2) = multiFlatten((   np.dot(inputs.T, d3), 
                                  d3.sum(0)[np.newaxis, :]  ))
    assert Dim2 == Dim

    ## E is the cost function (J from ufldl.standford.edu)
    ## df is the Jacobian of the cost function (the gradient of the cost function)
    return (E, df)



def backprop(VV, Dim, inputs, targets):
    W = [0]*4  #synaptic weight matrix
    hB = [0]*4  #hidden biases

    #### Un-Flatten all of our parameters from the 1-D array
    matrices = multiUnFlatten(VV, Dim)
    W[0]  = matrices[0]
    hB[0] = matrices[1]
    W[1]  = matrices[2]
    hB[1] = matrices[3]
    W[2]  = matrices[4]
    hB[2] = matrices[5]
    W[3]  = matrices[6]
    hB[3] = matrices[7]

    # Logistic activation function:
    actF = lambda x: 1./(1. + np.exp(-x))

    ## The four-layer neural network:
    layer0out = actF(   np.dot(inputs,    W[0]) + hB[0]) #numpy auto-tiles hB
    layer1out = actF(   np.dot(layer0out, W[1]) + hB[1]) #numpy auto-tiles hB
    layer2out = actF(   np.dot(layer1out, W[2]) + hB[2]) #numpy auto-tiles hB

    # final layer is softmax, not logsig:
    layer3out = np.exp( np.dot(layer2out, W[3]) + hB[3]) #numpy auto-tiles hB
    layer3out = layer3out / np.tile( layer3out.sum(1)[:, np.newaxis], (1,10) )

    # We use cross-entropy rather than squared error for our error function:
    # E for error
    ## the ufldl.stanford.edu link calls this the "Cost" function J.
    ## their version includes a weight decay term not listed here
    E = -(targets * np.log(layer3out)).sum(0).sum(0)
    # (... if both the targets and layer3out are normalized into probability distributions
    #  (as they are here) then this is the multi-class cross-entropy)

    # Classification error:
    # lowercase "delta" from Bishop, page 243, eq: 5.54:  
    #      δ_k = y_k - t_k
    # the ufldl.stanford.edu page uses a simlar nomenclature, but their final
    # layer is logsig, and they arrange the algebra a touch differently
    d3 = layer3out - targets

    # According to the backprop algorithm, we can assign this much error to these weights:
    deltaW3 = np.dot(layer2out.conj().T, d3)
    deltaHB3 = d3.sum(0)[np.newaxis, :]

    ## For backprop, we take the derivative actF acting on the
    ## input data.  
    ## For the Logistic function, the derivative of actF(x) 
    ## is actF(x) * (1 - actF(x))
    # dLogsig(x) is:  logsig(x) * (1-logsig(x))
    # Since we already have that value, we just re-use it
    # From Bishop, page 244, this is eq# 5.56:
    #   δ_j = h'(a_j) Σ ( w_kj * δ_k )
    # the ufldl.stanford.edu page:
    #  d_ = (W * d ) * f'()
    #     δ_j = h'(a_j) Σ ( w_kj * δ_k ) * f'(a_j)
    # =>  δ_j =  np.dot( w_k, δ_k ) .* f'(a_j)
    # =>  δ_j =  f'(a_j) .* np.dot( w_k, δ_k )
    # =>  δ_j =  f(a_j) * (1-f(a_j)) .* np.dot( δ_k, w_k )
    # which is what we have here:
    d2 = layer2out * (1-layer2out) * np.dot(d3, W[3].T)
    deltaW2 = np.dot(layer1out.T, d2)
    deltaHB2 = d2.sum(0)[np.newaxis, :]

    d1 = layer1out * (1-layer1out) * np.dot(d2, W[2].T)
    deltaW1 = np.dot(layer0out.T, d1)
    deltaHB1 = d1.sum(0)[np.newaxis, :]

    d0 = layer0out * (1-layer0out) * np.dot(d1, W[1].T)
    deltaW0 = np.dot(inputs.T, d0)
    deltaHB0 = d0.sum(0)[np.newaxis, :]

    ## Flatten the gradients into the same shape as VV:
    (df, Dim2) = multiFlatten(( deltaW0, deltaHB0, 
                                deltaW1, deltaHB1, 
                                deltaW2, deltaHB2, 
                                deltaW3, deltaHB3 ))
    assert Dim2 == Dim

    ## E is the cost function (J from ufldl.standford.edu)
    ## df is the Jacobian of the cost function (the gradient of the cost function)
    return (E, df)

