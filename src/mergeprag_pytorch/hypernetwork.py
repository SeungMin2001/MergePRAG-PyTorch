from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class HyperNetworkConfig:
    d_model: int = 512
    num_memory_vectors: int = 16
    dropout: float = 0.1


@dataclass(frozen=True)
class PassageMemory:
    """Memory key/value vectors produced for a passage or hop."""

    key: Tensor
    value: Tensor


class AttentivePooling(nn.Module):
    """Convert token-level passage states [B, T, d] into passage states [B, d]."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        scores = self.score(hidden_states).squeeze(-1)

        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        alpha = torch.softmax(scores, dim=1)
        return torch.sum(alpha.unsqueeze(-1) * hidden_states, dim=1)


class PassageMLP(nn.Module):
    """MLP_hyp(h) = ReLU(V * LayerNorm(ReLU(W * h)))."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, pooled_passage: Tensor) -> Tensor:
        h = self.relu(self.w(pooled_passage))
        h = self.layer_norm(h)
        h = self.dropout(h)
        return self.relu(self.v(h))


class MemoryProjection(nn.Module):
    """Project a passage vector into k memory keys and k memory values."""

    def __init__(self, d_model: int, num_memory_vectors: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_memory_vectors = num_memory_vectors
        self.key_projection = nn.Linear(d_model, num_memory_vectors * d_model)
        self.value_projection = nn.Linear(d_model, num_memory_vectors * d_model)

    def forward(self, passage_vector: Tensor) -> PassageMemory:
        batch_size = passage_vector.size(0)
        key = self.key_projection(passage_vector)
        value = self.value_projection(passage_vector)

        key = key.view(batch_size, self.num_memory_vectors, self.d_model)
        value = value.view(batch_size, self.num_memory_vectors, self.d_model)
        return PassageMemory(key=key, value=value)


class HyperNetwork(nn.Module):
    """Attentive pooling + MLP + K/V projection, matching the README study flow."""

    def __init__(self, config: HyperNetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.pooling = AttentivePooling(config.d_model)
        self.mlp = PassageMLP(config.d_model, config.dropout)
        self.projection = MemoryProjection(config.d_model, config.num_memory_vectors)

    def forward(self, passage_hidden_states: Tensor, attention_mask: Tensor | None = None) -> PassageMemory:
        pooled = self.pooling(passage_hidden_states, attention_mask)
        passage_vector = self.mlp(pooled)
        return self.projection(passage_vector)
