"""
Command Example: python model_unweighted.py music
"""

import sys
import torch
import torch.nn as nn
import pickle
from testing import test_model
from train_model import train_model
from utils import load_data, seed_everything, get_device

# experimental setup values
seed = 1234
batch_size = 100
domain = sys.argv[1]
weighted = "unweighted"
dataamount = "34000"

seed_everything(seed)
torch.set_printoptions(edgeitems=2)
print("Loading the data...")
train_loader, val_loader, test_loader = load_data(domain, batch_size)

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
device = get_device()

word2idx = pickle.load(open(f'dataset/' + domain + '/word2idx.pkl', 'rb'))


# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = n_layers
        self.dropout = nn.Dropout(drop_prob)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_size)  # 2 for bidirection, when dropout
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        embeds = self.embedding(x)

        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate the weighted/unweighted embeddings to Bi-LSTM
        lstm_out, (hidden, cell) = self.lstm(embeds, (h0, c0))

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        fc_out = self.fc(hidden)

        out = self.sigmoid(fc_out)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden

# training and testing the model
model = BiRNN(vocab_size=len(word2idx) + 1, output_size=1, embedding_dim=300, hidden_dim=512, n_layers=1).to(device)

train_model(model=model, batch_size=batch_size, train_loader=train_loader,
            val_loader=val_loader, device=device, clip=5,
            domain=domain, weighted=weighted, dataamount=dataamount, seed=seed)

# Loading the best model
model.load_state_dict(torch.load('models/' + domain + '/awsa_' + weighted + '_' + dataamount + '_' + str(seed) + '.pt'))

test_model(model, batch_size, test_loader, device, domain, dataamount, seed)
