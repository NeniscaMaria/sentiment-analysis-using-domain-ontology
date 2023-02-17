import numpy as np
import pickle

words = []
idx = 0
word2idx = {}
embeddings = []

with open('../resources/glove.840B.300d.txt', 'rb') as f:
    for l in f:
        try:
            line = l.decode().split()
            word = line[0]
            emb = np.array(line[1:]).astype(np.float64)
            words.append(word)
            word2idx[word] = idx
            idx += 1
            embeddings.append(emb)
        except Exception as e:
            print(e, word)
            pass

pickle.dump(words, open('840B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open('840B.300_idx.pkl', 'wb'))
pickle.dump(embeddings, open('840B.300_embs.pkl', 'wb'))
