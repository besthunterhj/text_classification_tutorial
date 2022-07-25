import torch
import torch.nn as nn


class RNNTextClassifier(nn.Module):

    def __init__(self, vocab_len: int, embedding_dim: int, hidden_dim: int, labels_len: int):
        super(RNNTextClassifier, self).__init__()
        # Init the attributes of the model(In other word, they are the hyper-parameters)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Init the layer of the RNN model
        self.embedding = nn.Embedding(vocab_len, self.embedding_dim)
        # Read the document of pytorch to finish this block
        self.rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # nonlinearity='relu'
        )
        self.linear = nn.Linear(
            self.hidden_dim, labels_len
        )

    # 5.2 Init the process of the model
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        The procedure of the forward propagation, telling what should do to the model
        :param x: the input of the model, its size must be [batch_size, text_len]
        :return:
        """

        embeddings = self.embedding(x)

        _, last_hidden = self.rnn(embeddings)

        last_hidden = last_hidden.squeeze(dim=0)

        y_hat = self.linear(last_hidden)

        return y_hat


class StackedRNNTextClassifier(nn.Module):
    def __init__(self, vocab_len: int, embedding_dim: int, hidden_dim: int, labels_len: int):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim * 6

        self.embedding = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            nonlinearity='relu',
            num_layers=3,
            bidirectional=True,
        )
        self.Linear = nn.Linear(in_features=self.hidden_dim, out_features=labels_len)

    def forward(self, x: torch.tensor) -> torch.tensor:

        embeddings = self.embedding(x)

        _, last_hidden = self.rnn(embeddings)

        last_hidden = torch.cat(last_hidden, dim=0)

        y_hat = self.linear(last_hidden)

        return y_hat
