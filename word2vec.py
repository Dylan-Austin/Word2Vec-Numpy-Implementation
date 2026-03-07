from numpy import ndarray
import numpy as np

class SG_Word2Vec(): # Skip gram
    def __init__(self, vocabulary_size : int, d_embedding : int) -> None:
        self.input_embeddings = np.random.randn(vocabulary_size, d_embedding)
        self.output_embeddings = np.random.randn(vocabulary_size, d_embedding)
    
    # center_word : (batch, 1), positive_context : (batch, 1), negative_samples : (batch, neg_samples)
    def forward(self, center_word : ndarray, positive_context : ndarray, negative_samples : ndarray) -> ndarray:
        center_embedding = self.input_embeddings[center_word]
        
        positive_embedding = self.output_embeddings[positive_context]   
        negative_embedding = self.output_embeddings[negative_samples]

        embeddings = np.concatenate((positive_embedding, negative_embedding),axis=-2)

        logits = np.sum(center_embedding * embeddings, axis=-1)


        positive_logit = logits[:, 0]
        negative_logits = logits[:, 1:]

        positive_loss = np.logaddexp(0, -positive_logit)
        negative_loss = np.sum(np.logaddexp(0, negative_logits), axis=-1)
        
        batch_loss = positive_loss + negative_loss
        loss = np.sum(batch_loss)

        return loss

if __name__ == "__main__":
    w2v = SG_Word2Vec(1000, 300)

    center_word = np.array([[2], [5], [3], [2]])
    positive_context = np.array([[7], [43], [12], [4]])
    negative_samples = np.array([[5, 3, 2], [5, 6, 7], [4, 3, 2], [9, 8, 1]])

    loss = w2v.forward(center_word, positive_context, negative_samples)
    print(loss)