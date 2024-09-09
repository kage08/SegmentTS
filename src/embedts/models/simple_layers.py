import torch.nn as nn

from .transformers.layers import TransformerAttn


class EmbedAttenSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention module to
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        rnn_out: int = 40,
        dim_out: int = 50,
        n_layers: int = 1,
        bidirectional: bool = False,
        attn=TransformerAttn,
        dropout=0.0,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param dim_metadata: Dimensions of metadata for all sequences
        param rnn_out: output dimension for rnn
        """
        super(EmbedAttenSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.attn_layer = attn(self.rnn_out, self.rnn_out, self.rnn_out)
        self.out_layer = [
            nn.Linear(in_features=self.rnn_out, out_features=self.dim_out),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward_mask(self, seqs, mask):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = latent_seqs
        latent_seqs = self.attn_layer.forward_mask(latent_seqs, mask)
        latent_seqs = latent_seqs.sum(0)
        out = self.out_layer(latent_seqs)
        return out

    def forward(self, seqs, metadata):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = self.attn_layer(latent_seqs).sum(0)
        out = self.out_layer(latent_seqs)
        return out
