import torch

from lptm.segment.scoring import ScoringModuleAddn, ScoringModuleMult

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_addn():
    embed_size = 5
    hidden_size = 10
    scoring = ScoringModuleAddn(embed_size, hidden_size).to(device)
    time_embeds = torch.randn(2, 3, embed_size).to(device)
    scores = scoring(time_embeds)
    assert scores.size() == (2, 3, 3)


def test_mult():
    embed_size = 5
    hidden_size = 10
    scoring = ScoringModuleMult(embed_size, hidden_size).to(device)
    time_embeds = torch.randn(2, 3, embed_size).to(device)
    scores = scoring(time_embeds)
    assert scores.size() == (2, 3, 3)
