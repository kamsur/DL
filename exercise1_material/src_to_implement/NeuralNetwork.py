#from Layers import FullyConnected 
import copy
class NeuralNetwork:
    
    def __init__(self, optimizer):
        self.optimizer=optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
    
    def forward(self, input_tensor=None, label_tensor=None):
        if input_tensor is None and label_tensor is None:
            self.input_tensor,self.label_tensor = self.data_layer.next()
        else:
            self.input_tensor,self.label_tensor = input_tensor,label_tensor
        prediction_tensor=self.input_tensor
        for layer in self.layers:
            prediction_tensor=layer.forward(prediction_tensor)
        return self.loss_layer.forward(prediction_tensor,self.label_tensor) 

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
        self.layers.append(layer)
    
    def train(self, iterations):
        for i in range(iterations):
            input_tensor,label_tensor = self.data_layer.next()
            self.loss.append(self.forward(input_tensor,label_tensor))
            self.backward(label_tensor)
            
    def test(self, input_tensor):
        prediction_tensor=input_tensor
        for layer in self.layers:
            prediction_tensor=layer.forward(prediction_tensor)
        return prediction_tensor
            
    