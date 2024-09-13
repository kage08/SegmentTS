import pytest
import torch

from lptm.segment.scoring import ScoringModuleAddn, ScoringModuleMult


@pytest.mark.parametrize("embed_size", [15])
@pytest.mark.parametrize("hidden_size", [10, 15])
@pytest.mark.parametrize("batch_size", [1, 3, 4])
@pytest.mark.parametrize("seq_len", [5, 1, 15])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_addn(
    embed_size: int, hidden_size: int, batch_size: int, seq_len: int, device: str
) -> None:
    scoring = ScoringModuleAddn(embed_size, hidden_size).to(device)
    time_embeds = torch.randn(batch_size, seq_len, embed_size).to(device)
    scores = scoring(time_embeds)
    assert scores.size() == (batch_size, seq_len, seq_len)


@pytest.mark.parametrize("embed_size", [15])
@pytest.mark.parametrize("hidden_size", [10, 15])
@pytest.mark.parametrize("batch_size", [1, 3, 4])
@pytest.mark.parametrize("seq_len", [5, 1, 15])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mult(
    embed_size: int, hidden_size: int, batch_size: int, seq_len: int, device: str
) -> None:
    scoring = ScoringModuleMult(embed_size, hidden_size).to(device)
    time_embeds = torch.randn(batch_size, seq_len, embed_size).to(device)
    scores = scoring(time_embeds)
    assert scores.size() == (batch_size, seq_len, seq_len)
