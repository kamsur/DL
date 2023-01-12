import numpy as np
from Layers import Base

class Flatten(Base.BaseLayer):
    
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, input_tensor):
        self.input_dimensions = input_tensor.shape
        b = input_tensor.shape[0]
        input_tensor1 = np.zeros((b, np.prod(input_tensor.shape[1:])))
        for i in range(b):
            input_tensor1[i] = input_tensor[i].flatten()
        return input_tensor1
    
    def backward(self, error_tensor):
        error_tensor1 = error_tensor.reshape(self.input_dimensions)
        return error_tensor1
