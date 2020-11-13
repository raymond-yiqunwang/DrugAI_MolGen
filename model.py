import sys
import torch
import torch.nn as nn

class MolGen(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, embed_size, nlayer, dropout=0.2):
        super(MolGen, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_size = embed_size
        self.nlayer = nlayer

        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, nlayer, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, data, hidden):
        seq_len = data.size(0)
        batch_size = data.size(1)
        net = self.encoder(data)
        output, hidden = self.rnn(net, hidden)
        output = self.decoder(output)

        return output, hidden

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.nlayer, batch_size, self.hidden_size),
                  torch.zeros(self.nlayer, batch_size, self.hidden_size))
        
        return hidden


