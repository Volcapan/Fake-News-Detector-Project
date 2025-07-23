import torch
from torch.utils.data import Dataset

class fakeNewsDataset(Dataset):
    def __init__(self, fakeNewsDF):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.samples = []

        for titleAndText in fakeNewsDF.news["titleAndText"]:
            tatTensor = torch.tensor(titleAndText)
            tatTensor = tatTensor.to(self.device)
            
            self.samples.append(tatTensor)

        self.labels = torch.tensor(fakeNewsDF.news["truthfulness"])
        self.labels = self.labels.to(self.device)
        
        self.size = len(self.samples)

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.samples[index], self.labels[index]