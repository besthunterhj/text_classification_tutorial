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

        :param vocab_size: the
        :param embedding_size:
        :param hidden_size:
        :param layer_num:
        :param dropout:
        :param label_num:
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

    def attention(self, bigru_output, final_state):
        """

        :param bigru_output: [batch_size, seq_len, 2 * layer_num * hidden_size]
        :param final_state: [batch_size, (1), 2 * layer_num * hidden_size]
        """

        final_state = final_state.permute(0, 2, 1)

        # the matrix of attention scores, which should be like [batch_size, seq_len, (1)]
        attn_weights = torch.matmul(input=bigru_output, other=final_state)

        # divide sqrt(dim), similar to the self-attention
        attn_weights = attn_weights / np.sqrt(bigru_output.shape[-1])

        # normalize the "attn_weights" by softmax, which could make each element belong to (0, 1)
        soft_attn_weights = F.softmax(input=attn_weights, dim=1)

        # compute the final attention weighted sequence representation (weighted_context)
        # bigru_output: [batch_size, 2 * layer_num * hidden_size, seq_len]
        bigru_output = bigru_output.permute(0, 2, 1)

        # weighted_context: [batch_size, 2 * layer_num * hidden_size, 1]
        weighted_context = torch.matmul(input=bigru_output, other=soft_attn_weights.unsqueeze(2))

        # return the 2-d vectors and the attention vector
        return weighted_context.squeeze(2), soft_attn_weights.data()

    def forward(self, input_ids: torch.Tensor):
        """

        """

