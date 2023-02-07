import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import os


# load the data from the csv file and perform a train-test-split
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        if name == 'data.csv':
            csv_path = os.path.join(root, name)
dataFrame = pd.read_csv(csv_path, sep=';')
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
train_dF, val_dF = train_test_split(dataFrame, test_size=0.20, random_state=40)
# TODO

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dl = t.utils.data.DataLoader(ChallengeDataset(train_dF, 'train'), batch_size=70, shuffle = True)
val_dl = t.utils.data.DataLoader(ChallengeDataset(val_dF, 'val'), batch_size=70)
# TODO

# create an instance of our ResNet model
resNet=model.ResNet()
# TODO

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
lossCrit=t.nn.CrossEntropyLoss()
# set up the optimizer (see t.optim)
optimizer=t.optim.Adam(resNet.parameters(),lr=1e-3,weight_decay=5*1e-3)
# create an object of type Trainer and set its early stopping criterion
trainer=Trainer(resNet,lossCrit,optimizer,train_dl,val_dl,cuda=True,early_stopping_patience=10)
# TODO

# go, go, go... call fit on trainer
res = trainer.fit(epochs=80)#TODO

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')