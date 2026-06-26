import torch
from torch import nn

from mergeprag_pytorch import (
    AttentivePooling,
    HyperNetwork,
    HyperNetworkConfig,
    MemoryInjector,
    build_spt_record,
    make_injection_hook,
    memory_cross_attention,
    orthogonal_merge,
)


def test_build_spt_record_from_hotpotqa_like_example() -> None:
    example = {
        "question": "q",
        "answer": "a",
        "context": {
            "title": ["Doc A", "Doc B"],
            "sentence": [["a0", "a1"], ["b0", "b1"]],
        },
        "supporting_facts": {"title": ["Doc B", "Doc A"], "sent_id": [1, 0]},
    }

    record = build_spt_record(example)

    assert record == {"question": "q", "answer": "a", "facts": ["b1", "a0"]}


def test_attentive_pooling_respects_padding_mask() -> None:
    pooling = AttentivePooling(d_model=3)
    hidden = torch.tensor([[[1.0, 2.0, 3.0], [100.0, 100.0, 100.0]]])
    mask = torch.tensor([[1, 0]])

    pooled = pooling(hidden, mask)

    assert torch.allclose(pooled, hidden[:, 0])


def test_hypernetwork_outputs_kv_memory_shapes() -> None:
    model = HyperNetwork(HyperNetworkConfig(d_model=8, num_memory_vectors=3, dropout=0.0))
    hidden = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)

    memory = model(hidden, mask)

    assert memory.key.shape == (2, 3, 8)
    assert memory.value.shape == (2, 3, 8)


def test_orthogonal_merge_adds_only_new_direction() -> None:
    existing = torch.tensor([[1.0, 0.0, 0.0]])
    update = torch.tensor([[1.0, 1.0, 0.0]])

    merged = orthogonal_merge(existing, update, eps=1e-8)

    assert torch.allclose(merged, torch.tensor([[1.0, 1.0, 0.0]]), atol=1e-5)


def test_memory_cross_attention_shape_and_weights() -> None:
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 3, 8)
    value = torch.randn(2, 3, 8)

    output, weights = memory_cross_attention(query, key, value, num_heads=2, return_weights=True)

    assert output.shape == query.shape
    assert weights.shape == (2, 2, 4, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 2, 4), atol=1e-6)


def test_injection_hook_supports_tuple_outputs() -> None:
    injector = MemoryInjector(d_model=8, num_heads=2, dropout=0.0)
    memory_key = torch.randn(1, 3, 8)
    memory_value = torch.randn(1, 3, 8)
    hook = make_injection_hook(injector, memory_key, memory_value)

    output = (torch.randn(1, 4, 8), torch.tensor(1.0))
    injected, aux = hook(nn.Identity(), tuple(), output)

    assert injected.shape == (1, 4, 8)
    assert aux.item() == 1.0
