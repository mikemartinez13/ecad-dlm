"""DiffuCoder entropy confidence shared by replay and decoding."""
import torch

def entropy_confidence(logits: torch.Tensor, temperature: float, top_p: float, top_p_logits) -> torch.Tensor:
    """DiffuCoder's 'entropy' confidence (negative entropy), without sampling a token."""

    if temperature > 0:
        logits = logits / temperature
    if top_p < 1:
        logits = top_p_logits(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.sum(probs * torch.log(probs + 1e-10), dim=-1)


