import numpy as np
import sys
sys.path.insert(0, r"C:\Users\anshu\OneDrive\Desktop\FAU\DL\exercise2_material\src_to_implement\Layers")
from Layers import Base

class Pooling(Base.BaseLayer):
    
    def __init__(self, stride_shape, pooling_shape):
        super(Pooling, self).__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
    
    def forward(self, input_tensor):
        input_tensor1 = None
        return input_tensor1
    
    def backward(self, error_tensor):
        error_tensor1 = None
        return error_tensor1