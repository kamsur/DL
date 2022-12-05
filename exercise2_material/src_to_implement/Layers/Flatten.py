import numpy as np
import sys
sys.path.insert(0, r"C:\Users\anshu\OneDrive\Desktop\FAU\DL\exercise2_material\src_to_implement\Layers")
import Base

class Flatten(Base.BaseLayer):
    
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, input_tensor):
        a = input_tensor.shape[0]
        b = input_tensor.shape[1]
        c = input_tensor.shape[2]
        d = input_tensor.shape[3]
        self.input_dimensions = (a, b, c, d)
        input_tensor1 = np.zeros((a, b*c*d))
        for i in range(a):
            input_tensor1[i] = input_tensor[i].flatten()
        return input_tensor1
    
    def backward(self, error_tensor):
        error_tensor1 = error_tensor.reshape(self.input_dimensions)
        return error_tensor1