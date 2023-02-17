import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
import random
import os


def load_data(domain, batch_size):
    train_sentences = pickle.load(open(f'dataset/' + domain + '/train_sentences.pkl', 'rb'))
    val_sentences = pickle.load(open(f'dataset/' + domain + '/val_sentences.pkl', 'rb'))
    test_sentences = pickle.load(open(f'dataset/' + domain + '/test_sentences.pkl', 'rb'))
    train_labels = pickle.load(open(f'dataset/' + domain + '/train_labels.pkl', 'rb'))
    val_labels = pickle.load(open(f'dataset/' + domain + '/val_labels.pkl', 'rb'))
    test_labels = pickle.load(open(f'dataset/' + domain + '/test_labels.pkl', 'rb'))
    train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
    val_data = TensorDataset(torch.from_numpy(val_sentences), torch.from_numpy(val_labels))
    test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def softmax(l):
    return np.exp(l) / np.sum(np.exp(l))


def get_device():
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        return torch.device("cuda:2")
    else:
        return torch.device("cpu")