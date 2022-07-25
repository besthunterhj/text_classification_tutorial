from typing import Dict

from torchtext.vocab import Vocab
from torchtext.vocab import vocab as vc
from collections.abc import Callable
from collections import Counter


def create_vocab(texts: list, tokenizer: Callable, min_freq: int, unknown_token: str = "<unk>",
                 unknown_index: int = 0) -> Vocab:

    all_tokens = [
        token
        for text in texts
        for token in tokenizer(text)
    ]

    tokens_counter = Counter(all_tokens)

    tokens_dict = dict(
        sorted(
            tokens_counter.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    vocab = vc(
        ordered_dict=tokens_dict,
        min_freq=min_freq
    )

    vocab.insert_token(
        token=unknown_token,
        index=unknown_index
    )

    vocab.set_default_index(index=unknown_index)

    return vocab


def create_labels_mapping(labels: list) -> Dict[str, int]:

    labels = list(set(labels))

    labels = sorted(labels, reverse=True)

    labels_mapping = {
        textual_label: index
        for index, textual_label in enumerate(labels)
    }

    return labels_mapping
