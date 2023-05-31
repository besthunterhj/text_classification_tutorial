import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BiGRUAttention(nn.Module):

    def __init__(
                self,
                vocab_size: int,
                embedding_size: int,
                hidden_size: int,
                layer_num: int,
                dropout: float,
                label_num: int
    ):
        """
        A multi-layers bidirectional GRU for text classification
        :param vocab_size: the length of the vocabulary
        :param embedding_size: the number of dimensions in the token embedding
        :param hidden_size: the number of dimensions from the output of this model
        :param layer_num: the number of layers of GRU
        :param dropout: the dropout rate
        :param label_num: the number of ground truth labels
        """
        super(BiGRUAttention, self).__init__()

        # the architecture of network
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.bigru = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=layer_num,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        # the Feed Forward Network (2-layers MLP)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2 * layer_num * hidden_size, out_features=4 * layer_num * hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=4 * layer_num * hidden_size, out_features=label_num)
        )

        self.hidden_size = hidden_size

    def attention(self, bigru_output, final_state):
        """
        implement the attention mechanism
        :param bigru_output: [batch_size, seq_len, 2 * layer_num * hidden_size]
        :param final_state: [ 2 * layer_num, batch_size, hidden_size]
        """

        # change the dimension permutation of "final_state" and named it for "hidden"
        # hidden: [batch_size, 2 * layer_num * hidden_size, 1]
        hidden = final_state.view(-1, self.hidden_size * 2, 1)
        # final_state = final_state.permute(0, 2, 1)

        # the matrix of attention scores, which should be like [batch_size, seq_len, (1)]
        attn_weights = torch.matmul(input=bigru_output, other=hidden)

        # divide sqrt(dim), similar to the self-attention
        attn_weights = attn_weights / np.sqrt(bigru_output.shape[-1])

        # normalize the "attn_weights" by softmax, which could make each element belong to (0, 1)
        soft_attn_weights = F.softmax(input=attn_weights, dim=1)

        # compute the final attention weighted sequence representation (weighted_context)
        # bigru_output: [batch_size, 2 * layer_num * hidden_size, seq_len]
        bigru_output = bigru_output.permute(0, 2, 1)

        # weighted_context: [batch_size, 2 * layer_num * hidden_size, 1]
        weighted_context = torch.matmul(input=bigru_output, other=soft_attn_weights)

        # return the 2-d vectors and the attention vector
        return weighted_context.squeeze(2), soft_attn_weights.squeeze(2)

    def forward(self, input_ids: torch.Tensor):
        """
        the forward process of this multi-layers bi-GRU
        :param input_ids: the ids sequences: [batch_size, seq_len, 1]
        """

        # current_embeddings: [batch_size, seq_len, embedding_dim]
        current_embeddings = self.embedding(input_ids)

        # input to the gru model
        # gru_outputs: [batch_size, seq_len, 2 * layer_num * hidden_size]
        # final_state: [2 * layer_num, batch_size, hidden_size]
        gru_outputs, final_state = self.bigru(current_embeddings)

        # atten_weighted_outputs: [batch_size, 2 * layer_num * hidden_size]
        # attention_scores: [batch_size, seq_len]
        atten_weighted_outputs, attention_scores = self.attention(gru_outputs, final_state)

        # classification
        logits = self.classifier(atten_weighted_outputs)

        return logits


