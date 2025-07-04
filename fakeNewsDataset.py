import torch
from torch.utils.data import Dataset

class fakeNewsDataset(Dataset):
    def __init__(self, news):
        print()