import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass
    
    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        loss = np.sum(-np.log(prediction_tensor[label_tensor==1] + np.finfo(float).eps))
        return loss
    
    def backward(self, label_tensor):
        error_tensor = -label_tensor/(self.prediction_tensor + np.finfo(float).eps)
        return error_tensor
