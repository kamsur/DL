import numpy as np
from Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super(SoftMax, self).__init__()
    
    def forward(self, input_tensor):
        shifted_input_tensor=input_tensor-np.max(input_tensor) 
        sum1 = np.sum(np.exp(shifted_input_tensor), axis = 1)
        self.output_tensor = np.divide(np.exp(shifted_input_tensor),np.expand_dims(sum1, axis=1))
        return self.output_tensor
    
    def backward(self, error_tensor):
        sum2 = np.sum(error_tensor*self.output_tensor, axis = 1)
        sum2 = np.expand_dims(sum2, axis=1)
        error_tensor1 = self.output_tensor*(error_tensor - sum2)
        return error_tensor1