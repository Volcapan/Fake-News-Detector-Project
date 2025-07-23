import torch
from torch.nn.utils import rnn

def collate_func(batch):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    sequences, labels = zip(*batch)
    seqLengths = []

    for sequence in sequences:
        seqLengths.append(len(sequence))
        
    seqLengths = torch.tensor(seqLengths)
    seqLengths = seqLengths.to(device)
    seqLengths, slIndices = torch.sort(seqLengths, descending=True)

    sequences = torch.tensor(sequences)
    sequences = sequences.to(device)
    sequences = rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    sequences = sequences[slIndices]
    sequences = rnn.pack_padded_sequence(sequences, seqLengths.cpu(), batch_first=True)

    labels = torch.tensor(labels)[slIndices]
    labels = labels.to(device)

    return sequences, labels