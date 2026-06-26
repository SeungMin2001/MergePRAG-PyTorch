import torch

from mergeprag_pytorch import (
    HyperNetwork,
    HyperNetworkConfig,
    MemoryInjector,
    build_spt_record,
    merge_memory_sequence,
)


def main() -> None:
    example = {
        "question": "Where was the author of Hamlet born?",
        "answer": "Stratford-upon-Avon",
        "context": {
            "title": ["Hamlet", "William Shakespeare"],
            "sentence": [
                ["Hamlet is a tragedy written by William Shakespeare."],
                ["William Shakespeare was born in Stratford-upon-Avon."],
            ],
        },
        "supporting_facts": {
            "title": ["Hamlet", "William Shakespeare"],
            "sent_id": [0, 0],
        },
    }

    record = build_spt_record(example)
    print("SPt facts:", record["facts"])

    d_model = 32
    num_passages = len(record["facts"])
    token_len = 6
    passage_states = torch.randn(num_passages, token_len, d_model)
    passage_mask = torch.ones(num_passages, token_len, dtype=torch.bool)

    hypernetwork = HyperNetwork(
        HyperNetworkConfig(d_model=d_model, num_memory_vectors=4, dropout=0.0)
    )
    memory = hypernetwork(passage_states, passage_mask)

    merged_key = merge_memory_sequence([memory.key[0], memory.key[1]])
    merged_value = merge_memory_sequence([memory.value[0], memory.value[1]])

    hidden_states = torch.randn(1, 5, d_model)
    injector = MemoryInjector(d_model=d_model, num_heads=4, dropout=0.0)
    injected = injector(hidden_states, merged_key.unsqueeze(0), merged_value.unsqueeze(0))

    print("memory key shape:", tuple(memory.key.shape))
    print("injected hidden shape:", tuple(injected.shape))


if __name__ == "__main__":
    main()
