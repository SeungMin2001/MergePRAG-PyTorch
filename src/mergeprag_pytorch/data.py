from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _first_existing(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"none of these keys exist: {keys}")


def make_supporting_passages(example: Mapping[str, Any]) -> list[str]:
    """Build SPt from a HotpotQA-style example.

    The README describes SPt as the passage list obtained by matching
    supporting_facts.title/sent_id against context.title/sentence.
    This function keeps that idea but handles both `sentence` and `sentences`
    field names because HotpotQA loaders differ slightly by version.
    """

    context = example["context"]
    supporting_facts = example["supporting_facts"]

    context_titles: Sequence[str] = _first_existing(context, "title", "titles")
    context_sentences: Sequence[Sequence[str]] = _first_existing(
        context, "sentence", "sentences"
    )
    support_titles: Sequence[str] = _first_existing(supporting_facts, "title", "titles")
    support_ids: Sequence[int] = _first_existing(
        supporting_facts, "sent_id", "sent_ids", "sentence_id"
    )

    title_to_index = {title: idx for idx, title in enumerate(context_titles)}
    passages: list[str] = []

    for title, sent_id in zip(support_titles, support_ids):
        context_index = title_to_index.get(title)
        if context_index is None:
            continue

        sentences = context_sentences[context_index]
        if 0 <= int(sent_id) < len(sentences):
            passage = sentences[int(sent_id)].strip()
            if passage:
                passages.append(passage)

    return passages


def build_spt_record(example: Mapping[str, Any]) -> dict[str, Any]:
    """Return the question/answer/facts record used before reasoning-chain work."""

    return {
        "question": example.get("question", ""),
        "answer": example.get("answer", ""),
        "facts": make_supporting_passages(example),
    }
