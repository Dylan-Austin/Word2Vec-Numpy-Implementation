from numpy import ndarray
import numpy as np

def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class SG_Word2Vec(): # Skip gram
    def __init__(self, vocabulary_size: int, d_embedding: int) -> None:
        self.input_embeddings = np.random.randn(vocabulary_size, d_embedding) * 0.01
        self.output_embeddings = np.random.randn(vocabulary_size, d_embedding) * 0.01

    # expects (batch, 1), (batch, 1), (batch, neg_samples)
    def training_step(self, center_word: ndarray, positive_context: ndarray, negative_samples: ndarray, lr: float) -> float:
        center_embedding = self.input_embeddings[center_word]
        
        positive_embedding = self.output_embeddings[positive_context]   
        negative_embeddings = self.output_embeddings[negative_samples]

        embeddings = np.concatenate((positive_embedding, negative_embeddings),axis=-2)

        logits = np.sum(center_embedding * embeddings, axis=-1)

        positive_logit = logits[:, 0]
        negative_logits = logits[:, 1:]

        positive_loss = np.logaddexp(0, -positive_logit)
        negative_loss = np.sum(np.logaddexp(0, negative_logits), axis=-1)

        # backprop step
        positive_loss_gradient = (sigmoid(positive_logit) - 1).reshape(-1, 1, 1)
        negative_loss_gradient = np.expand_dims(sigmoid(negative_logits), -1)

        # positive loss gradients
        center_gradients = positive_loss_gradient * positive_embedding
        # negative loss gradients
        center_gradients += np.sum(negative_loss_gradient * negative_embeddings, axis=-2, keepdims=True)

        positive_gradients = positive_loss_gradient * center_embedding
        negative_gradients = negative_loss_gradient * center_embedding

        # SGD
        np.add.at(self.input_embeddings, center_word, -lr * center_gradients)
        np.add.at(self.output_embeddings, positive_context, -lr * positive_gradients)
        np.add.at(self.output_embeddings, negative_samples, -lr * negative_gradients)

        batch_loss = positive_loss + negative_loss
        loss = np.mean(batch_loss)

        return loss