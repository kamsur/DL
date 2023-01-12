import numpy as np
from Layers import Base

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()
        pass

    def forward(self, input_tensor):
        self.sig = 1/(1+np.exp(-input_tensor))
        return self.sig

    def backward(self, error_tensor):
        return error_tensor * self.sig * (1-self.sig)
