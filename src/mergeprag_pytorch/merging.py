from __future__ import annotations

import torch
from torch import Tensor


def _orthogonal_merge_2d(existing: Tensor | None, update: Tensor, eps: float) -> Tensor:
    if existing is None:
        return update
    if existing.shape != update.shape:
        raise ValueError(f"shape mismatch: existing={existing.shape}, update={update.shape}")

    # Memory tensors are [k, d]. Projection is easier in [d, k],
    # where each memory vector becomes a basis column in d-dimensional space.
    basis = existing.transpose(0, 1)
    candidate = update.transpose(0, 1)

    num_vectors = basis.size(1)
    feature_dim = basis.size(0)
    gram = basis.transpose(0, 1) @ basis
    gram = gram + eps * torch.eye(num_vectors, dtype=basis.dtype, device=basis.device)
    projection = basis @ torch.linalg.inv(gram) @ basis.transpose(0, 1)

    identity = torch.eye(feature_dim, dtype=basis.dtype, device=basis.device)
    orthogonal_component = (identity - projection) @ candidate
    merged = basis + orthogonal_component
    return merged.transpose(0, 1)


def orthogonal_merge(existing: Tensor | None, update: Tensor, eps: float = 1e-6) -> Tensor:
    """Merge a new memory bank while keeping only its component outside the old span.

    Supports [k, d] for one example and [B, k, d] for a batch.
    """

    if update.ndim == 2:
        return _orthogonal_merge_2d(existing, update, eps)

    if update.ndim == 3:
        if existing is None:
            return update
        if existing.shape != update.shape:
            raise ValueError(f"shape mismatch: existing={existing.shape}, update={update.shape}")
        merged = [
            _orthogonal_merge_2d(old_memory, new_memory, eps)
            for old_memory, new_memory in zip(existing, update, strict=True)
        ]
        return torch.stack(merged, dim=0)

    raise ValueError("memory tensors must be [k, d] or [B, k, d]")


def merge_memory_sequence(memory_sequence: list[Tensor], eps: float = 1e-6) -> Tensor:
    """Merge memories from passages or hops in order."""

    if not memory_sequence:
        raise ValueError("memory_sequence must contain at least one tensor")

    merged: Tensor | None = None
    for memory in memory_sequence:
        merged = orthogonal_merge(merged, memory, eps)
    return merged
