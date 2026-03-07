import re
import numpy as np
from numpy import ndarray

class Tokeniser():
    def __init__(self):
        self.vocab = {}

    def fit_vocab(self, dataset) -> None:
        tokens = set()
        for line in dataset:
            for token in line.split():
                tokens.add(token.lower())

        self.vocab = {word:i for i, word in enumerate(tokens)}        

    def tokenise(self, dataset) -> ndarray:
        tokens = []
        for line in dataset:
            tokens.extend([self.vocab[word.lower()] for word in line.split()])
        return np.array(tokens)