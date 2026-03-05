from numpy import ndarray
import numpy as np

class Word2Vec():
    def __init__(self, vocabulary_size : int, d_embedding : int) -> None:
        # embedding matrix
        self.embeddings = np.random.rand(vocabulary_size, d_embedding)

        # linear layer
        self.weights = np.random.rand(d_embedding, vocabulary_size)
        self.biases = np.random.rand(vocabulary_size)
    
    def forward(self, x : ndarray) -> ndarray:
        # extract the word embeddings
        x = self.embeddings[x]

        # average them
        x = x.mean(axis=-2)

        # apply the linear layer
        x = x @ self.weights + self.biases

        return x