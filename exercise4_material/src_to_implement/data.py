from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
from pandas import DataFrame


class ChallengeDataset(Dataset):
    def __init__(self, data:DataFrame, mode:str):
        super().__init__()
        self.data = data
        self.mode = mode
        self._transform = tv.transforms.Compose()
    # TODO implement the Dataset class according to the description
    #pass

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transforms_list):
        self._transform = tv.transforms.Compose(transforms_list)
