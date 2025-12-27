import numpy as np
from gensim.models import KeyedVectors

def build_embedding_matrix(tokenizer, embedding_dim, glove_path, emoji2vec_path, vocab_size):
    embedding_index = {}

    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.rstrip().split()
            embedding_index[values[0]] = np.asarray(values[1:], dtype='float32')

    emoji_vectors = KeyedVectors.load_word2vec_format(emoji2vec_path, binary=False)
    for e in emoji_vectors.key_to_index:
        embedding_index[e] = emoji_vectors[e]

    embedding_matrix = np.random.normal(0, 1, (vocab_size, embedding_dim))

    for word, idx in tokenizer.word_index.items():
        if idx < vocab_size and word in embedding_index:
            embedding_matrix[idx] = embedding_index[word]

    return embedding_matrix
