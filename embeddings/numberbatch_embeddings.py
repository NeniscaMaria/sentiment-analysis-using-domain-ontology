import numpy as np
import pickle

words = []
idx = 0
word2idx = {}
embeddings = []

with open(f'../resources/numberbatch-en.txt', 'rb') as f:
    for l in f:
        if idx == 0:
            idx += 1
            continue
        line = l.decode().split()
        word = line[0]
        emb = np.array(line[1:]).astype(np.float64)
        embeddings.append(emb)
        words.append(word)
        word2idx[word] = idx - 1
        idx += 1

pickle.dump(words, open(f'cn_nb.300_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'cn_nb.300_idx.pkl', 'wb'))
pickle.dump(embeddings, open(f'cn_nb.300_embs.pkl', 'wb'))
