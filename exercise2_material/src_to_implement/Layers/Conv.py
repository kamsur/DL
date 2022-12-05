import numpy as np
from scipy import signal
import sys
sys.path.insert(0, r"C:\Users\anshu\OneDrive\Desktop\FAU\DL\exercise2_material\src_to_implement\Layers")
import Base

class Conv(Base.BaseLayer):
    
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super(Conv, self).__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        
    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
        
    @property
    def gradient_bias(self):
        return self._gradient_bias
    
    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value
        
    def forward(self, input_tensor):
        pass
    
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def optimizer_bias(self):
        return self._optimizer_bias
    
    @optimizer_bias.setter
    def optimizer_bias(self, value):
        self._optimizer_bias = value
        
    def backward(self, error_tensor):
        pass
    
    def initialize(self, weights_initializer, bias_initializer):
        pass