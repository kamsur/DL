import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda and t.cuda.is_available():
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if (self._cuda and t.cuda.is_available()) else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()
        # -propagate through the network
        pred = self._model(x)
        # -calculate the loss
        loss = self._crit(pred, y.float())
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss.item()
        #TODO
        
        
    
    def val_test_step(self, x, y):
        
        # predict
        pred = self._model(x)
        # propagate through the network and calculate the loss and predictions
        loss = self._crit(pred, y.float())
        # return the loss and the predictions
        return loss.item(), pred
        #TODO
        
    def train_epoch(self):
        # set training mode
        self._model = self._model.train()
        # iterate through the training set
        loss = 0
        for img, label in self._train_dl:
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda and t.cuda.is_available():
                img = img.to('cuda')
                label = label.to('cuda')
            else:
                img = img.to('cpu')
                label = label.to('cpu')
        # perform a training step
            loss = loss + self.train_step(x=img, y=label)        
        # calculate the average loss for the epoch and return it
        avg_loss = loss / len(self._train_dl)
        return avg_loss
        #TODO
    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model = self._model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with t.no_grad():
            total_loss = 0
            preds = None
            labels = None 
            # iterate through the validation set
            for img, label in self._val_test_dl:
            # transfer the batch to the gpu if given
                if self._cuda and t.cuda.is_available():
                    img = img.to('cuda')
                    label = label.to('cuda')
                else:
                    img = img.to('cpu')
                    label = label.to('cpu')
            # perform a validation step
                loss, pred = self.val_test_step(img, label)
                total_loss = total_loss + loss
            # save the predictions and the labels for each batch
                if preds is None and labels is None:
                    labels = label
                    preds = pred
                else:
                    labels = t.cat((labels, label), dim=0)
                    preds = t.cat((preds, pred), dim=0)
            # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
            avg_loss=total_loss / len(self._val_test_dl)
            self.f1_score = f1_score(t.squeeze(labels.cpu()), t.squeeze(preds.cpu().round()), average='weighted')
            print("F1 score={},Val_loss={}".format(self.f1_score,avg_loss))
            # return the loss and print the calculated metrics
            return avg_loss
        #TODO
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        val_losses = []
        epoch_cntr = 0
        patience_cntr=0
        val_loss_min=None
        f1_max=0
        self.f1_scores=[]
        #TODO
        
        while True:
      
            # stop by epoch number
            if epoch_cntr == epochs:
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            epoch_cntr += 1
            train_loss = self.train_epoch()
            val_loss = self.val_test()
            # append the losses to the respective lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            self.f1_scores.append(self.f1_score)
            if val_loss_min is None:
                val_loss_min=val_loss
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if (len(val_losses)>0 and val_loss <= val_loss_min) or (self.f1_score>=f1_max):
                self.save_checkpoint(epoch_cntr)
                patience_cntr=0
                f1_max=self.f1_score
                val_loss_min=val_loss
            elif (len(val_losses) >1 and val_loss > 1.02 * val_losses[-2]):
                patience_cntr += 1
            print("Epoch counter={},Patience counter={},f1_max={}\n".format(epoch_cntr,patience_cntr,f1_max))
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if epoch_cntr==epochs or (self._early_stopping_patience>0 and patience_cntr==self._early_stopping_patience):
                return train_losses,val_losses
            # return the losses for both training and validation
        #TODO