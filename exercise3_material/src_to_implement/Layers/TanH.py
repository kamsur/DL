import numpy as np
from Layers import Base

class TanH(Base.BaseLayer):
    def __init__(self):
        super(TanH, self).__init__()
        self.tanh=None
        pass

    def forward(self, input_tensor):
        self.tanh = np.tanh(input_tensor)
        return self.tanh

    def backward(self, error_tensor):
        return error_tensor * (1 - self.tanh**2)
