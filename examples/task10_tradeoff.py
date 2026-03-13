"""Task 10: Demonstrate a compression-vs-robustness trade-off."""
from __future__ import annotations

import tempfile
from pathlib import Path

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual, wordlevel_multilingual
from abctokz.eval.metrics import fertility, mean_tokens_per_sentence, unk_rate

CORPUS_LINES = [
    "jana gana mana adhinayaka jaya he bharata bhagya vidhata",
    "punjab sindhu gujarata maratha dravida utkala vanga",
    "jan gan man adhinayak jay he bharat bhagya vidhata",
    "जन गण मन अधिनायक जय हे भारत भाग्य विधाता",
    "पंजाब सिंधु गुजरात मराठा द्राविड़ उत्कल बंगा",
    "hello world tokenizer benchmark deterministic",
] * 60

EVAL_TEXTS = [
    "jana gana mana adhinayaka jaya he",
    "जन गण मन अधिनायक जय हे",
    "hello world tokenizer benchmark",
    "xylophonically quantumflux",  # rare unseen words
    "hello 😀 world",  # unseen symbol class
]


def evaluate(name: str, config, corpus_path: str) -> None:
    tokenizer = Tokenizer.from_config(config)
    tokenizer.train([corpus_path], config)

    encodings = [tokenizer.encode(t) for t in EVAL_TEXTS]
    word_counts = [len(t.split()) for t in EVAL_TEXTS]

    fert = fertility(encodings, word_counts)
    unk = unk_rate(encodings)
    mtps = mean_tokens_per_sentence(encodings)

    print(f"\n=== {name} ===")
    print(f"vocab_size={tokenizer.get_vocab_size()}")
    print(f"fertility={fert:.3f}")
    print(f"unk_rate={unk:.3f}")
    print(f"mean_tokens_per_sentence={mtps:.3f}")

    print("sample encodings:")
    for text, enc in zip(EVAL_TEXTS, encodings):
        print(f"- {text!r}")
        print(f"  tokens={enc.tokens}")
        print(f"  ids={enc.ids}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "task10_corpus.txt"
        corpus_path.write_text("\n".join(CORPUS_LINES), encoding="utf-8")

        evaluate("BPE (vocab_size=200)", bpe_multilingual(vocab_size=200), str(corpus_path))
        evaluate("WordLevel (vocab_size=200)", wordlevel_multilingual(vocab_size=200), str(corpus_path))


if __name__ == "__main__":
    main()
