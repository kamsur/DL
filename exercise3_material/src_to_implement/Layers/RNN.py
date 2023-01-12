import numpy as np
import copy
from Layers import Base
from Layers import FullyConnected
from Layers import TanH

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable=True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._memorize = False
        self._weights=None
        self._gradient_weights=None
        self._optimizer=None
        self.h_t=None
        self.last_seq_h=None
        self.time_size=None
        self.input_tensor = None
        self.FC_h = FullyConnected.FullyConnected(hidden_size + input_size, hidden_size)

        self.tanH=TanH.TanH()
        self.FC_y = FullyConnected.FullyConnected(hidden_size, output_size)
        
    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        self.FC_y.initialize(weights_initializer, bias_initializer)
        self.FC_h.initialize(weights_initializer, bias_initializer)
        self.weights = self.FC_h.weights
        self.gradient_weights = self.FC_h.gradient_weights

    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.time_size=input_tensor.shape[0]
        if self.memorize:
            if self.h_t is None:
                self.h_t = np.zeros((self.time_size + 1, self.hidden_size))
            else:
                self.h_t[0] = self.last_seq_h
        else:
            self.h_t = np.zeros((self.time_size + 1, self.hidden_size))

        y_t = np.zeros((self.time_size, self.output_size))
        for t in range(self.time_size):
            h = self.h_t[t][np.newaxis, :]
            x = input_tensor[t][np.newaxis, :]
            xtilda_t = np.concatenate((h, x), axis = 1)
            self.h_t[t+1] = self.tanH.forward(self.FC_h.forward(xtilda_t))
            y_t[t] = (self.FC_y.forward(self.h_t[t + 1][np.newaxis, :]))
        
        self.last_seq_h = self.h_t[-1]

        return y_t

    def backward(self, error_tensor):
        self.Ex_out=np.zeros((self.time_size, self.input_size))
        Eh = np.zeros((1, self.hidden_size))
        for t in reversed(range(self.time_size)):
            self.FC_y.input_tensor = np.hstack((self.h_t[t+1], 1))[np.newaxis, :]
            Ey = self.FC_y.backward(error_tensor[t][np.newaxis, :])
            grad_yh = Eh + Ey
            self.tanH.tanh=self.h_t[t+1]
            grad_h = self.tanH.backward(grad_yh)
            self.FC_h.input_tensor = np.hstack((self.h_t[t], self.input_tensor[t],1))[np.newaxis, :]
            Exh = self.FC_h.backward(grad_h)
            Eh = Exh[:, 0:self.hidden_size]
            Ex = Exh[:, self.hidden_size:(self.hidden_size + self.input_size + 1)]
            self.Ex_out[t] = Ex
            self.weights = self.FC_h.weights
            self.gradient_weights = self.FC_h.gradient_weights

        if self.optimizer is not None:
            weights_y = self.optimizer.calculate_update(self.FC_y.weights, self.FC_y.gradient_weights)
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.FC_y.weights = weights_y
            self.FC_h.weights = self.weights
        return self.Ex_out


