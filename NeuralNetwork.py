# Note that the code is specific to the architecture of this neural network

import numpy as np
import matplotlib as plot
import pandas as pd

# x is a numpy array of lists. 
# Each list has the number of study and sleep hours per student.

# y is a numpy array of lists. 
# Each list has the score for each student based on the study and sleep hour

x = np.array(([3,5], [5,1],[10,2]), dtype = float)
# y = np.array(([75],[82],[93]), dtype = float)

# Normalizing x and y values. Using a maximum score of 100.
x = x/np.amax(x, axis = 0)
# y = y/100


class NeuralNet(object):
    def __init__(self, inputnode, hiddenode, outputnode):
        '''
        The constructor method defining the 
        structure of the neural network
        '''
        self.inputnode = inputnode
        self.hiddenode = hiddenode
        self.outputnode = outputnode
        
    def feedforward(self, x):
        '''
        x = an array
        '''
        # Weight parameters
        self.WIH = np.random.randn(self.inputnode, \
                                   self.hiddenode)
        self.WHO = np.random.randn (self.hiddenode, \
                                    self.outputnode)
        
        # Matrix multiplication
        self.z2 = np.dot(x, self.WIH)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.WHO)
        yHat = self.sigmoid(self.z3)
        
        return yHat
        
    def sigmoid(self, z):
        '''
        The activation function
        '''
        return 1/(1+np.exp(-z))

# Test parameters
NN = NeuralNet(2,3,1)
yHat = NN.feedforward(x)

print(yHat)