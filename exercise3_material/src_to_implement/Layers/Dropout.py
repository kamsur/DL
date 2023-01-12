
import numpy as np
from Layers import Base

class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.dropped = 0

    def forward(self, input_tensor):
        if self.testing_phase ==False:

            self.dropped = np.random.uniform(0, 1, size =(input_tensor.shape[0], input_tensor.shape[1])) < self.probability
            output_tensor = (input_tensor * self.dropped)/self.probability

            return output_tensor
        
        elif self.testing_phase == True:
            return input_tensor

    def backward(self, error_tensor):
        return (error_tensor * self.dropped)/self.probability
    
