import numpy as np

import sys
sys.path.insert(0, r"C:\Users\anshu\OneDrive\Desktop\FAU\DL\exercise3_material\src_to_implement\Layers")
import Base

class TanH(Base.BaseLayer):
    def __init__(self):
        super(TanH, self).__init__()
        pass

    def forward(self, input_tensor):
        self.tanh = np.tanh(input_tensor)
        return self.tanh

    def backward(self, error_tensor):
        return error_tensor * (1 - self.tanh**2)