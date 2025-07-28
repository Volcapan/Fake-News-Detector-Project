import torch
from torch.nn.utils import rnn
import getDevice

def collate_func(batch):
    device = getDevice.get_device_func()
    sequences, labels = zip(*batch)
    seqLengths = []

    for sequence in sequences:
        seqLengths.append(len(sequence))
        
    seqLengths = torch.tensor(seqLengths)
    seqLengths, slIndices = torch.sort(seqLengths, descending=True)

    sequences = torch.tensor(sequences)
    sequences = rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    sequences = sequences[slIndices]
    sequences = rnn.pack_padded_sequence(sequences, seqLengths.cpu(), batch_first=True)
    sequences = sequences.to(device)

    labels = torch.tensor(labels)[slIndices]
    labels = labels.to(device)

    return sequences, labels