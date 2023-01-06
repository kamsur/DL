import numpy as np

import sys
sys.path.insert(0, r"C:\Users\anshu\OneDrive\Desktop\FAU\DL\exercise3_material\src_to_implement\Layers")
import Base

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()
        pass

    def forward(self, input_tensor):
        self.sig = 1/(1+np.exp(-input_tensor))
        return self.sig

    def backward(self, error_tensor):
        return error_tensor * self.sig * (1-self.sig)