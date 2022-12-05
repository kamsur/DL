import numpy as np

import sys
sys.path.insert(0,r"C:\Users\anshu\OneDrive\Desktop\FAU\DL\exercise2_material\src_to_implement\Optimization")
sys.path.insert(0, r"C:\Users\anshu\OneDrive\Desktop\FAU\DL\exercise2_material\src_to_implement\Layers")
import Optimizers
import Base, Initializers



class FullyConnected(Base.BaseLayer):
    
    def __init__(self, input_size, output_size):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, size = (self.input_size+1, self.output_size))
        self.trainable = True
        self._optimizer = None
        self._gradient_weights=None
        
    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.vstack((self.weights, self.bias))
    
    @property
    def optimizer(self):
        return self._optimizer 
    
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
    
    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights= value
    
    def forward(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        input_tensor1 = np.concatenate((input_tensor, np.ones([self.batch_size, 1])), axis=1)
        self.input_tensor=input_tensor1
        self.output_tensor = np.dot(input_tensor1, self.weights)
        return self.output_tensor
        
    def backward(self, error_tensor):
        self.error_tensor1 = np.dot(error_tensor, self.weights.T)
        gradient_tensor=np.dot(self.input_tensor.T, error_tensor)
        self.gradient_weights=gradient_tensor
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return self.error_tensor1[:,:-1]
    