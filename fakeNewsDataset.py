import torch
from torch.utils.data import Dataset

class fakeNewsDataset(Dataset):
    def __init__(self, news):
        self.samples = news.drop("truthfulness", axis=1)
        self.labels = news["truthfulness"]
        self.len = len(self.news)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.samples[index], self.labels[index]