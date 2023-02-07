from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv
from pandas import DataFrame

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    def __init__(self, data:DataFrame, mode:str):
        super().__init__()
        self.data = data
        self.mode = mode
        self._transform = tv.transforms.Compose(transforms=[tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)])
    # TODO implement the Dataset class according to the description
    #pass

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transforms_list=[tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)]):
        self._transform = tv.transforms.Compose(transforms=transforms_list)

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        filename, isCrack, isInactive = self.data.iloc[index]
        img = imread(Path(filename))
        img = gray2rgb(img)
        transformer = self.transform
        img = transformer(img)
        return (img, torch.tensor([isCrack, isInactive]))
