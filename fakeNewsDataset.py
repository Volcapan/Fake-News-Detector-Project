import torch
from torch.utils.data import Dataset
from torch.nn.utils import rnn

class fakeNewsDataset(Dataset):
    def __init__(self, fakeNewsDF):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.samples = []
        sampleLengths = []

        for titleAndText in fakeNewsDF.news["titleAndText"]:
            self.samples.append(torch.tensor(titleAndText))
            sampleLengths.append(len(titleAndText))

        sampleLengths = torch.tensor(sampleLengths)
        sampleLengths, slIndices = torch.sort(sampleLengths, descending=True)
        

        self.samples = rnn.pad_sequence(self.samples, batch_first=True, padding_value=0)
        self.samples = self.samples[slIndices]
        #self.samples = rnn.pack_padded_sequence(self.samples, sampleLengths.cpu(), batch_first=True)
        self.samples = self.samples.to(self.device)

        self.labels = torch.tensor(fakeNewsDF.news["truthfulness"])
        self.labels = self.labels.to(self.device)
        
        self.batchSize = len(self.samples)
        self.seqSize = len(self.samples[0])

    def __len__(self):
        return self.batchSize
    
    def __getitem__(self, index):
        return self.samples[index], self.labels[index]