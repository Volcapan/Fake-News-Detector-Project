from torch import nn

class fakeNewsNN(nn.Module):
    def __init__(self, vocabulary_size):
        embedding_dim = 10
        lstm_hidden_dim = 15
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim, padding_idx=0)
        self.nnModel = nn.Sequential(nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, batch_first=True),
                                     nn.Linear(in_features=lstm_hidden_dim, out_features=1))
    
    def forward(self, input):
        print("In progress")