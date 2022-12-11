import numpy as np
#import sys
#sys.path.insert(0, r"C:\BABU\FAU\FAU coding\DL\exercise2_material\src_to_implement\Layers")
from Layers import Base
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
                if len(self.convolution_shape)==3:
                    ch = ch[::self.stride_shape[0], ::self.stride_shape[1]]
                elif len(self.convolution_shape)==2:
                    ch = ch[::self.stride_shape[0]]
                output_tensor[img, output_ch]= np.add(ch,self.bias[output_ch])
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
        E_n_1 = np.zeros_like(self.input_tensor)
        weights_temp = np.copy(self.weights)

        if len(self.convolution_shape)==3:
            gradW_temp = np.zeros((error_tensor.shape[0], self.weights.shape[0], self.weights.shape[1],
                                              self.weights.shape[2], self.weights.shape[3]))
            padded_input = []
            for img in range(self.input_tensor.shape[0]):
                channels = []
                for input_ch in range(self.input_tensor.shape[1]):
                    channels.append(np.pad(self.input_tensor[img, input_ch], ((self.convolution_shape[1]//2, self.convolution_shape[1]//2),
                                                                                 (self.convolution_shape[2]//2,
                                                                                  self.convolution_shape[2]//2)), mode='constant'))
                    if self.convolution_shape[2]%2 ==0:
                        channels[input_ch] = channels[input_ch][:,:-1]
                    if self.convolution_shape[1]%2 ==0:
                        channels[input_ch] = channels[input_ch][:-1,:]

                channels = np.stack(channels, axis=0)
                channels.tolist()
                padded_input.append(channels)
            padded_input = np.stack(padded_input, axis=0)
            for img in range(error_tensor.shape[0]):
                for output_ch in range(error_tensor.shape[1]):
                    upsampled = signal.resample(error_tensor[img, output_ch], error_tensor[img, output_ch].shape[0] * self.stride_shape[0], axis=0)
                    upsampled = signal.resample(upsampled, error_tensor[img, output_ch].shape[1] * self.stride_shape[1], axis=1)
                    upsampled = upsampled[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]
                    if self.stride_shape[1] > 1:
                        for i, row in enumerate(upsampled):
                            for j in range(len(row)):
                                if j % self.stride_shape[1] != 0:
                                    row[j] = 0
                    if self.stride_shape[0] > 1:
                        for i, row in enumerate(upsampled):
                            if i % self.stride_shape[0] != 0:
                                for j in range(len(row)):
                                    row[j] = 0

                    for input_ch in range(self.input_tensor.shape[1]):
                        gradW_temp[img, output_ch, input_ch] = signal.correlate(padded_input[img, input_ch], upsampled, mode='valid')
            self.gradient_weights = gradW_temp.sum(axis=0)
        
        if len(self.convolution_shape)==3:
            weights_temp = np.transpose(weights_temp, (1,0,2,3))
        elif len(self.convolution_shape)==2:
            weights_temp = np.transpose(weights_temp, (1,0,2))
        
        for img in range(error_tensor.shape[0]):
            for kernel in range(weights_temp.shape[0]):
                ch_E_n_1 = None
                for ch in range(weights_temp.shape[1]):
                    if len(self.convolution_shape) == 3:
                        upsampled = signal.resample(error_tensor[img, ch], error_tensor[img, ch].shape[0] * self.stride_shape[0], axis=0)
                        upsampled = signal.resample(upsampled, error_tensor[img, ch].shape[1] * self.stride_shape[1], axis=1)
                        upsampled = upsampled[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]
                        if self.stride_shape[1] > 1:
                            for i, row in enumerate(upsampled):
                                for j in range(len(row)):
                                    if j % self.stride_shape[1] != 0:
                                        row[j] = 0
                        if self.stride_shape[0] > 1:
                            for i, row in enumerate(upsampled):
                                for j in range(len(row)):
                                    if i % self.stride_shape[0] != 0:
                                        row[j] = 0

                    elif len(self.convolution_shape) == 2:
                        upsampled = signal.resample(error_tensor[img, ch], error_tensor[img, ch].shape[0] * self.stride_shape[0], axis=0)
                        upsampled = upsampled[:self.input_tensor.shape[2]]
                        if self.stride_shape[0] > 1:
                            for i in range(len(upsampled)):
                                if i % self.stride_shape[0] != 0:
                                    upsampled[i] = 0
                    if ch_E_n_1 is None:
                        ch_E_n_1=signal.convolve(upsampled, weights_temp[kernel, ch], mode='same')
                    else:
                        ch_E_n_1=ch_E_n_1+(signal.convolve(upsampled, weights_temp[kernel, ch], mode='same'))
                E_n_1[img, kernel] = ch_E_n_1

        if len(self.convolution_shape)==3:
            self.gradient_bias = np.sum(error_tensor, axis=(0,2,3))
        elif len(self.convolution_shape)==2:
            self.gradient_bias = np.sum(error_tensor, axis=(0,2))

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        if self.optimizer_bias:
            self.bias = self.optimizer_bias.calculate_update(self.bias, self.gradient_bias)

        return E_n_1


    
    def initialize(self, weights_initializer, bias_initializer):
        if len(self.convolution_shape) == 3:
            self.weights = weights_initializer.initialize((self.num_kernels, self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2]),
                                                          self.convolution_shape[0]*self.convolution_shape[1]* self.convolution_shape[2],
                                                          self.num_kernels*self.convolution_shape[1]* self.convolution_shape[2])
            self.bias = bias_initializer.initialize((self.num_kernels), 1, self.num_kernels)
            self.bias = self.bias[-1]


        elif len(self.convolution_shape) == 2:
            self.weights = weights_initializer.initialize((self.num_kernels, self.convolution_shape[0], self.convolution_shape[1]),
                                                          self.convolution_shape[0]*self.convolution_shape[1],
                                                          self.num_kernels*self.convolution_shape[1])
            self.bias = bias_initializer.initialize((1, self.num_kernels), 1, self.num_kernels)
            self.bias = self.bias[-1]

