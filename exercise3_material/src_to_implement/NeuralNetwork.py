#from Layers import FullyConnected
import copy
class NeuralNetwork:
    
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = []
        self.loss_layer = []
        
    @property
    def phase(self):
        return self._phase
    @phase.setter
    def phase(self, value):
        self._phase = value
    
    def forward(self, input_tensor=None, label_tensor=None):
        if input_tensor is None and label_tensor is None:
            self.input_tensor,self.label_tensor = self.data_layer.next()
        else:
            self.input_tensor,self.label_tensor = input_tensor,label_tensor
        prediction_tensor=self.input_tensor
        regularization_loss = 0
        for layer in self.layers:
            prediction_tensor=layer.forward(prediction_tensor)
            try:
                if layer.optimizer:
                    if layer.optimizer.regularizer:
                        regularization_loss += layer.optimizer.regularizer.norm(layer.weights)
            except(AttributeError):
                pass
        loss = self.loss_layer.forward(prediction_tensor,self.label_tensor)
        self.loss.append(loss+regularization_loss)
        return loss + regularization_loss

    def backward(self, label_tensor=None):
        if label_tensor is None:
            error_tensor=self.loss_layer.backward(self.label_tensor)
        else:
            error_tensor=self.loss_layer.backward(label_tensor) 
        for layer in self.layers[::-1]:
            error_tensor=layer.backward(error_tensor)
        return error_tensor
    
    def append_layer(self, layer):
        if layer.trainable == True:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)
    
    def train(self, iterations):
        self.testing_phase = False
        for i in range(iterations):
            input_tensor,label_tensor = self.data_layer.next()
            self.loss.append(self.forward(input_tensor,label_tensor))
            self.backward(label_tensor)
            
    def test(self, input_tensor):
        self.testing_phase = True
        prediction_tensor=input_tensor
        for layer in self.layers:
            layer.testing_phase = True
            prediction_tensor=layer.forward(prediction_tensor)
        return prediction_tensor
            
