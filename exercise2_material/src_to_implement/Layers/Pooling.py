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
        for i in range(input_tensor.shape[0]):
            for j in range(input_tensor.shape[1]):
                for pi in range(self.output_tensor.shape[2]):
                    for pj in range(self.output_tensor.shape[3]):
                        self.output_tensor[i, j,pi ,pj ] = np.max(input_tensor[i, j, self.stride_shape[0]*pi:pi*self.stride_shape[0]+self.pooling_shape[0], self.stride_shape[1]*pj:pj*self.stride_shape[1]+self.pooling_shape[1]])
        self.output_tensor = self.output_tensor.astype(int)
        return self.output_tensor
    
    def backward(self, error_tensor):
        self.max_pos = np.zeros(self.input_tensor.shape)
        for i in range(self.input_tensor.shape[0]):
            for j in range(self.input_tensor.shape[1]):
                for pi in range(error_tensor.shape[2]):
                    for pj in range(error_tensor.shape[3]):
                        self.max_pos[i, j, self.stride_shape[0]*pi:pi*self.stride_shape[0]+self.pooling_shape[0], self.stride_shape[1]*pj:pj*self.stride_shape[1]+self.pooling_shape[1]] = np.where(np.max(self.input_tensor[i, j, self.stride_shape[0]*pi:pi*self.stride_shape[0]+self.pooling_shape[0], self.stride_shape[1]*pj:pj*self.stride_shape[1]+self.pooling_shape[1]])==self.input_tensor[i, j, self.stride_shape[0]*pi:pi*self.stride_shape[0]+self.pooling_shape[0], self.stride_shape[1]*pj:pj*self.stride_shape[1]+self.pooling_shape[1]], self.input_tensor[i, j, self.stride_shape[0]*pi:pi*self.stride_shape[0]+self.pooling_shape[0], self.stride_shape[1]*pj:pj*self.stride_shape[1]+self.pooling_shape[1]], 0)
        self.max_pos = self.max_pos.astype(int)
        return self.max_pos
