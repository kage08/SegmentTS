from typing import Tuple
import einops
import torch
from torch import Tensor
import torch.nn as nn
from .transformers.layers import CrossAttention
from .simple_layers import EmbedAttenSeq


class LatentEmbedder(nn.Module):
    """
    Converts a batch of embeding to attention over latent embeddings
    """

    def __init__(self, latent_dim: int, in_dim: int, num_latent: int) -> None:
        """
        Args:
            latent_dim: Dimensionality of latent vector
            in_dim: Dimensionality of input vector
            num_latent: Number of latent embeddings
        """
        super(LatentEmbedder, self).__init__()
        self.latent_dim = latent_dim
        self.in_dim = in_dim
        self.num_latent = num_latent

        self.latent_embeds = nn.Parameter(
            torch.randn(num_latent, latent_dim), requires_grad=True
        )

        self.embedder = CrossAttention(in_dim, latent_dim, latent_dim)

    def forward(self, seqs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            seqs: [batch_size, seq_len, in_dim]
        """
        latent_embeds = einops.repeat(
            self.latent_embeds, "l d -> b l d", b=seqs.shape[0]
        )
        output, attention = self.embedder(seqs, latent_embeds)
        return output, attention.squeeze()


class SPIN_Encoder(nn.Module):
    """
    Takes a batch of time-series segments; embeds using a GRU+SA and pass through
    SPIN Embedder

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        latent_dim: int,
        *,
        bidirectional: bool = True,
        dropout: float = 0.0,
        num_latent=64
    ) -> None:
        """
        Args:
            in_dim: Dimensionality of input vector seq. features
            out_dim: Dimensionality of output vector
            latent_dim: Dimensionality of latent vector
            bidirectional: Whether to use bidirectional GRU
            dropout: Dropout rate
            num_latent: Number of latent embeddings
        """
        super(SPIN_Encoder, self).__init__()

        self.seq_embedder = EmbedAttenSeq(
            in_dim, out_dim, bidirectional=bidirectional, dropout=dropout
        )
        self.latent_embedder = LatentEmbedder(latent_dim, out_dim, num_latent)

    def forward(self, seq_segments: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            seq_segments: [batch_size, num_segs, seq_len, in_dim]
        """
        seq_segments_ct = einops.rearrange(
            seq_segments, "b s l d -> (b s) l d"
        )  # [batch_size * num_segs, seq_len, in_dim]
        seq_embeds, _ = self.seq_embedder(
            seq_segments_ct
        )  # [batch_size * num_segs, out_dim]
        seq_embeds = einops.rearrange(
            seq_embeds, "(b s) d -> b s d", b=seq_segments.shape[0]
        )  # [batch_size, num_segs, out_dim]
        latent_embeds, attention = self.latent_embedder(
            seq_embeds
        )  # [batch_size, num_segs, latent_dim], [batch_size, num_segs, num_latent]
        return latent_embeds, attention
