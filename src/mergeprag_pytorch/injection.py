from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def _combine_heads(x: Tensor) -> Tensor:
    batch_size, _, seq_len, head_dim = x.shape
    num_heads = x.size(1)
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)


def memory_cross_attention(
    query: Tensor,
    memory_key: Tensor,
    memory_value: Tensor,
    num_heads: int = 8,
    return_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Attention from model hidden states Q to HyperNetwork memory K/V."""

    if query.size(-1) != memory_key.size(-1) or memory_key.shape != memory_value.shape:
        raise ValueError("query, memory_key, and memory_value must share the same d_model")
    if query.size(-1) % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    head_dim = query.size(-1) // num_heads
    q = _split_heads(query, num_heads)
    k = _split_heads(memory_key, num_heads)
    v = _split_heads(memory_value, num_heads)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1)
    output = _combine_heads(weights @ v)

    if return_weights:
        return output, weights
    return output


class MemoryInjector(nn.Module):
    """Small trainable adapter used inside a forward hook for memory injection."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden_states: Tensor, memory_key: Tensor, memory_value: Tensor) -> Tensor:
        query = self.query_projection(hidden_states)
        key = self.key_projection(memory_key)
        value = self.value_projection(memory_value)
        injected = memory_cross_attention(query, key, value, self.num_heads)
        injected = self.output_projection(injected)

        # The scalar gate starts at 0.5 after sigmoid and can learn how strongly
        # the merged passage memory should modify the frozen base model state.
        mixed = hidden_states + torch.sigmoid(self.gate) * self.dropout(injected)
        return self.layer_norm(mixed)


def freeze_module(module: nn.Module) -> None:
    """Freeze a base model before training only the HyperNetwork/injector."""

    for parameter in module.parameters():
        parameter.requires_grad = False


def make_injection_hook(
    injector: MemoryInjector,
    memory_key: Tensor,
    memory_value: Tensor,
) -> Callable[[nn.Module, tuple[object, ...], Tensor | tuple[Tensor, ...]], Tensor | tuple[Tensor, ...]]:
    """Create a PyTorch forward hook for transformer blocks.

    Hugging Face layers sometimes return a tensor and sometimes a tuple whose
    first item is hidden_states. This hook supports both forms.
    """

    def hook(
        module: nn.Module,
        inputs: tuple[object, ...],
        output: Tensor | tuple[Tensor, ...],
    ) -> Tensor | tuple[Tensor, ...]:
        del module, inputs

        if isinstance(output, tuple):
            hidden_states = output[0]
            injected = injector(
                hidden_states,
                memory_key.to(device=hidden_states.device, dtype=hidden_states.dtype),
                memory_value.to(device=hidden_states.device, dtype=hidden_states.dtype),
            )
            return (injected, *output[1:])

        return injector(
            output,
            memory_key.to(device=output.device, dtype=output.dtype),
            memory_value.to(device=output.device, dtype=output.dtype),
        )

    return hook
