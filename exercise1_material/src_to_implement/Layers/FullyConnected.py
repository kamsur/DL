import numpy as np

#import sys
#sys.path.insert(0,r"C:\BABU\FAU\FAU coding\exercise1_material\src_to_implement\Layers")
#sys.path.insert(0, r"C:\BABU\FAU\FAU coding\exercise1_material\src_to_implement\Optimization")
#from Optimization import Optimizers
from Layers import Base

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0,1,(self.input_size+1,self.output_size))
        self.trainable = True
        self._optimizer = None
        self._gradient_weights=None
    
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
        self.input_tensor= np.concatenate((input_tensor, np.ones((input_tensor.shape[0], 1))), axis=1)
        self.output_tensor = np.dot(self.input_tensor,self.weights)
        return self.output_tensor
        
    def backward(self, error_tensor):
        self.error_tensor=error_tensor
        E_n_1 = np.dot(self.error_tensor,self.weights.T)
        gradient_tensor=np.dot(self.input_tensor.T,self.error_tensor)
        self.gradient_weights=gradient_tensor
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, gradient_tensor)
        return E_n_1[:,:-1]
    
