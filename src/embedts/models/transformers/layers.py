import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    Positional encoding as described in "Attention is all you need"

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).

    From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class CrossAttention(nn.Module):
    """
    Cross-Attention over set of sequences


    Args:
        dim_in1: dimension of input sequence
        dim_in2: dimension of input sequence
        dim_out: dimension of output sequence
        n_heads: number of heads
        dropout: dropout rate
    """

    def __init__(
        self,
        dim_in_query: int,
        dim_in_key: int,
        dim_out: int,
        n_heads: int = 1,
        dropout: float = 0.0,
        *,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super(CrossAttention, self).__init__()
        self.dim_in_key = dim_in_key
        self.dim_in_query = dim_in_query
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim_out
        assert self.hidden_dim % n_heads == 0
        self.dim_out = dim_out
        self.n_heads = n_heads
        self.dropout = dropout

        self.in_layer_key = nn.Linear(
            in_features=dim_in_key, out_features=self.hidden_dim
        )
        self.in_layer_query = nn.Linear(
            in_features=dim_in_query, out_features=self.hidden_dim
        )
        self.in_layer_value = nn.Linear(
            in_features=dim_in_key, out_features=self.dim_out
        )

    def forward(self, query: Tensor, key: Tensor):
        """

        Args:
            query: Tensor, shape [batch_size, seq_len_q, dim_in_query]
            key: Tensor, shape [batch_size, seq_len_k, dim_in_key]

        Returns:
            output: Tensor, shape [batch_size, seq_len_q, dim_out]
            attention: Tensor, shape [n_heads, batch_size, seq_len_q, seq_len_k]
        """

        query = self.in_layer_query(query)  # [batch_size, seq_len_q, hidden_dim]
        key = self.in_layer_key(key)  # [batch_size, seq_len_k, hidden_dim]
        value = self.in_layer_value(key)  # [batch_size, seq_len_k, dim_out]

        # Split into n_heads
        query = query.view(query.shape[0], query.shape[1], self.n_heads, -1).permute(
            2, 0, 1, 3
        )  # [n_heads, batch_size, seq_len_q, hidden_dim/n_heads]
        key = key.view(key.shape[0], key.shape[1], self.n_heads, -1).permute(
            2, 0, 1, 3
        )  # [n_heads, batch_size, seq_len_k, hidden_dim/n_heads]
        value = value.view(value.shape[0], value.shape[1], self.n_heads, -1).permute(
            2, 0, 1, 3
        )  # [n_heads, batch_size, seq_len_k, dim_out/n_heads]

        # Compute attention

        attention = torch.einsum("hbid,hbjd->hbij", query, key) / math.sqrt(
            self.hidden_dim
        )  # [n_heads, batch_size, seq_len_q, seq_len_k]
        attention = F.softmax(
            attention, dim=-1
        )  # [n_heads, batch_size, seq_len_q, seq_len_k]
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        # Compute output
        output = torch.einsum(
            "hbij,hbjd->hbid", attention, value
        )  # [n_heads, batch_size, seq_len_q, dim_out/n_heads]
        output = output.permute(
            1, 2, 0, 3
        ).contiguous()  # [batch_size, seq_len_q, n_heads, dim_out/n_heads]
        output = output.view(
            output.shape[0], output.shape[1], -1
        )  # [batch_size, seq_len_q, dim_out]

        return output, attention


class TransformerAttn(nn.Module):
    """
    Module that calculates self-attention weights using transformer like attention
    """

    def __init__(
        self, dim_in: int = 40, value_dim: int = 40, key_dim: int = 40
    ) -> None:
        """
        param dim_in: Dimensionality of input sequence
        param value_dim: Dimension of value transform
        param key_dim: Dimension of key transform
        """
        super(TransformerAttn, self).__init__()
        self.value_layer = nn.Linear(dim_in, value_dim)
        self.query_layer = nn.Linear(dim_in, value_dim)
        self.key_layer = nn.Linear(dim_in, key_dim)

    def forward(self, seq: Tensor):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.softmax(weights, -1)
        return (weights @ keys).transpose(1, 0)
