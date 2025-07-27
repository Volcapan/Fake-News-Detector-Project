import torch
from torch.utils.data import Dataset
import getDevice

class fakeNewsDataset(Dataset):
    def __init__(self, dataframe):
        self.device = getDevice.get_device_func()
        self.samples = []

        for titleAndText in dataframe["titleAndText"]:
            tatTensor = torch.tensor(titleAndText)
            tatTensor = tatTensor.to(self.device)
            
            self.samples.append(tatTensor)

        self.labels = torch.tensor(dataframe["truthfulness"].to_numpy())
        self.labels = self.labels.to(self.device)
        
        self.size = len(self.samples)

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.samples[index], self.labels[index]