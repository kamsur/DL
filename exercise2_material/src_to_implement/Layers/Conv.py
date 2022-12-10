import numpy as np
import sys
sys.path.insert(0, r"C:\BABU\FAU\FAU coding\DL\exercise2_material\src_to_implement\Layers")
import Base
from scipy import signal

class Conv(Base.BaseLayer):
    
    def __init__(self, stride_shape=np.random.uniform(0,1,1)[0], convolution_shape=np.random.uniform(0,1,2)[0], num_kernels=np.random.uniform(0,1,1)[0]):
        super(Conv, self).__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        if len(convolution_shape)==3:
            self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1], convolution_shape[2])
        elif len(convolution_shape)==2:
            self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1])
        self.bias = np.random.uniform(0,1,num_kernels)
        self._optimizer=None
        self._gradient_weights=None
        self._gradient_bias=None
        self._optimizer_bias=None
    

        
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
        self.input_tensor = input_tensor
        if len(self.convolution_shape)==3:
            output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, int(np.ceil(input_tensor.shape[2]/self.stride_shape[0])),
                               int(np.ceil(input_tensor.shape[3]/self.stride_shape[1]))))
        elif len(self.convolution_shape)==2:
            output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, int(np.ceil(input_tensor.shape[2]/self.stride_shape[0]))))
        for img in range(input_tensor.shape[0]):
            for output_ch in range(self.weights.shape[0]):
                ch=None
                for input_ch in range(self.weights.shape[1]):
                    if ch is None:
                        ch=signal.correlate(input_tensor[img, input_ch], self.weights[output_ch, input_ch], mode='same')
                    else:
                        ch=ch+(signal.correlate(input_tensor[img, input_ch], self.weights[output_ch, input_ch], mode='same'))
                #ch=np.stack(ch,axis=0)
                #ch=ch.sum(axis=0)
                if len(self.convolution_shape)==3:
                    ch = ch[::self.stride_shape[0], ::self.stride_shape[1]]
                elif len(self.convolution_shape)==2:
                    ch = ch[::self.stride_shape[0]]
                output_tensor[img, output_ch]= ch+self.bias[output_ch]
        return output_tensor

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
        if len(self.convolution_shape) == 3:
            self.weights = weights_initializer.initialize((self.num_kernels, self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2]),
                                                          self.convolution_shape[0]*self.convolution_shape[1]* self.convolution_shape[2],
                                                          self.num_kernels*self.convolution_shape[1]* self.convolution_shape[2])
            self.bias = bias_initializer.initialize((self.num_kernels), 1, self.num_kernels)

        elif len(self.convolution_shape) == 2:
            self.weights = weights_initializer.initialize((self.num_kernels, self.convolution_shape[0], self.convolution_shape[1]),
                                                          self.convolution_shape[0]*self.convolution_shape[1],
                                                          self.num_kernels*self.convolution_shape[1])
            self.bias = bias_initializer.initialize((1, self.num_kernels), 1, self.num_kernels)
