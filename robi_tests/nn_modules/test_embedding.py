import torch
from cs336_basics.nn_modules.embedding import Embedding


def test_embedding():
    num_embeddings = 6
    embedding_dim = 16
    my_module = Embedding(num_embeddings=num_embeddings,embedding_dim=embedding_dim)
    indices = torch.randint(0, num_embeddings, size=(2, 2, 2))
    print(my_module(indices).shape)
