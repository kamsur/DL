import numpy as np
#import sys
#sys.path.insert(0, r"C:\Users\anshu\OneDrive\Desktop\FAU\DL\exercise2_material\src_to_implement\Layers")
import Base

class Pooling(Base.BaseLayer):
    
    def __init__(self, stride_shape = np.random.uniform(0,1,1)[0], pooling_shape=np.random.uniform(0,1,1)[0]):
        super(Pooling, self).__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.zeros((input_tensor.shape[0], input_tensor.shape[1], (input_tensor.shape[2]-self.pooling_shape[0])//self.stride_shape[0] + 1, (input_tensor.shape[3]-self.pooling_shape[1])//self.stride_shape[1] + 1))
        self.max_pos = np.zeros(self.output_tensor.shape)
        for i in range(input_tensor.shape[0]):
            for j in range(input_tensor.shape[1]):
                for pi in range(self.output_tensor.shape[2]):
                    for pj in range(self.output_tensor.shape[3]):
                        a = self.stride_shape[0]*pi
                        b = self.stride_shape[1]*pj
                        self.output_tensor[i, j,pi ,pj] = np.max(input_tensor[i, j, a:a+self.pooling_shape[0], b:b+self.pooling_shape[1]])
                        self.max_pos[i, j, pi, pj] = np.argmax(input_tensor[i, j, a:a+self.pooling_shape[0], b:b+self.pooling_shape[1]])
        self.max_pos = self.max_pos.astype(int)
        return self.output_tensor
    
    def backward(self, error_tensor):
        error = np.zeros(self.input_tensor.shape)
        for i in range(self.input_tensor.shape[0]):
            for j in range(self.input_tensor.shape[1]):
                for pi in range(error_tensor.shape[2]):
                    for pj in range(error_tensor.shape[3]):
                        a, b = np.unravel_index(self.max_pos[i, j,pi,pj], (self.pooling_shape[0], self.pooling_shape[1]))        
                        if error[i, j, a+self.stride_shape[0]*pi, b+self.stride_shape[1]*pj] != 0.0:
                            error[i, j,a+self.stride_shape[0]*pi, b+self.stride_shape[1]*pj] += error_tensor[i, j, pi, pj]
                        else:
                            error[i, j,a+self.stride_shape[0]*pi, b+self.stride_shape[1]*pj] = error_tensor[i, j, pi, pj]
        return error
