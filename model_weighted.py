"""
Command Example: python model_weighted.py music
"""

import sys
import torch
import torch.nn as nn
import pickle
import json
from testing import test_model
from train_model import train_model
from utils import load_data, seed_everything, get_device

# experimental setup values
seed = 1234
batch_size = 100
domain = sys.argv[1]
weighted = "weighted"
dataamount = "50000"

seed_everything(seed)

print(weighted + " - " + dataamount + " - " + str(seed))

torch.set_printoptions(edgeitems=2)

train_loader, val_loader, test_loader = load_data(domain, batch_size)

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
device = get_device()

# For Weighted Word Embeddings 
with open("./ontology/" + domain + "/scores.json", "r") as f:
    scores = json.load(f)

word2idx = pickle.load(open(f'dataset/' + domain + '/word2idx.pkl', 'rb'))
idx2word = pickle.load(open(f'dataset/' + domain + '/idx2word.pkl', 'rb'))
cn_words = pickle.load(open(f'embeddings/cn_nb.300_words.pkl', 'rb'))
cn_word2idx = pickle.load(open(f'embeddings/cn_nb.300_idx.pkl', 'rb'))
cn_embs = pickle.load(open(f'embeddings/cn_nb.300_embs.pkl', 'rb'))


# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):

    def __init__(self, vocab_size, target_vocab, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = n_layers
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(drop_prob)
        tot_at = 0

        # Initialization of matrices
        scores_matrix = torch.ones((vocab_size, 1))
        weights_matrix = torch.ones((vocab_size, embedding_dim))
        for v in target_vocab:
            # initialize weights_matrix with conceptnet embeddings
            try:
                if v in ['_PAD', '_UNK']:
                    weights_matrix[word2idx[v]] = torch.from_numpy(cn_embs[0])
                else:
                    weights_matrix[word2idx[v]] = torch.from_numpy(cn_embs[cn_word2idx[v]])
            except:
                pass
            # initialize scores_matrix with aspect scores
            if v in scores.keys():
                tot_at = tot_at + 1
                scores_matrix[word2idx[v], 0] = scores[v]

        print("vocab_size = " + str(vocab_size) + " --- total aspect terms = " + str(tot_at))

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = torch.nn.Parameter(weights_matrix)
        self.aspect_scores = nn.Embedding(vocab_size, 1)
        self.aspect_scores.weight = torch.nn.Parameter(scores_matrix)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        embeds = self.embedding(x)
        # Multiplying the embeddings with the aspect scores from the aspect score layer
        scores = self.aspect_scores(x)
        scores1 = scores.repeat(1, 1, self.embedding_dim)
        embeds = embeds * scores1

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
model = BiRNN(vocab_size=len(word2idx) + 1, target_vocab=word2idx.keys(), output_size=1,
              embedding_dim=300, hidden_dim=512, n_layers=1).to(device)

lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_model(model=model, batch_size=batch_size, train_loader=train_loader,
            val_loader=val_loader, device=device, clip=5,
            domain=domain, weighted=weighted, dataamount=dataamount, seed=seed)

# Loading the best model
model.load_state_dict(torch.load('models/' + domain + '/awsa_' + weighted + '_' + dataamount + '_' + str(seed) + '.pt'))

test_model(model, batch_size, test_loader, device, domain, dataamount, seed)
