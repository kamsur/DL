import numpy as np

class Constant:
    def __init__(self, const_val = 0.1):
        self.const_val = const_val
        
    def initialize(self, weights_shape, fan_in, fan_out):
        self.weight = np.zeros((fan_in, fan_out)) + self.const_val
        return self.weight

class UniformRandom:
    
    def initialize(self, weights_shape, fan_in, fan_out):
        self.weight = np.random.uniform(0, 1, size = (fan_in, fan_out))
        return self.weight

class Xavier:
    
    def initialize(self, weights_shape, fan_in, fan_out):
        self.weight = np.random.normal(0, np.sqrt(2/(fan_out + fan_in)), weights_shape)
        return self.weight

class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        self.weight = np.random.normal(0, np.sqrt(2/fan_in), weights_shape)
        return self.weight