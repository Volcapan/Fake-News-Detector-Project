from torch import nn
from torch.nn.utils import rnn

class fakeNewsNN(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()

        embedding_dim = 150
        lstm_hidden_dim = 200

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=lstm_hidden_dim,
                            batch_first=True)
        self.linear = nn.Linear(in_features=lstm_hidden_dim,
                                out_features=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        padded_seqs, seq_lengths = rnn.pad_packed_sequence(input, batch_first=True)
        padded_embedded_seqs = self.embedding(padded_seqs)
        padded_embedded_seqs = rnn.pack_padded_sequence(padded_embedded_seqs, seq_lengths.cpu(), batch_first=True)

        _, (hidden_layers, _) = self.lstm(padded_embedded_seqs)
        
        return self.sigmoid(self.linear(hidden_layers[-1]))
