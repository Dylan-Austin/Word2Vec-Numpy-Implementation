from numpy import ndarray
import numpy as np

class Word2Vec():
    def __init__(self, vocabulary_size : int, d_embedding : int) -> None:
        self.embeddings = np.random.randn(vocabulary_size, d_embedding)
    
    # takes shape (batch, 2) -> (batch, 1)
    # expects index of word in vocab
    def forward(self, x : ndarray) -> ndarray:
        x = self.embeddings[x] # (batch, 2, d_embedding)

        # take dot product of target and context vectors
        x = np.sum(x[..., 0, :] * x[..., 1, :], axis=-1, keepdims=True) # (batch, 1)

        return x
    
if __name__ == "__main__":
    w2v = Word2Vec(1000, 300)
    
    output = w2v.forward(np.array([[2, 3], [1, 2]]))
    print(output.shape)