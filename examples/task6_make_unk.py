"""Task 6: Trigger and analyze <unk> behavior across model families."""
from __future__ import annotations

import tempfile
from pathlib import Path

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual, unigram_multilingual, wordlevel_multilingual
from abctokz.eval.metrics import unk_rate

CORPUS_LINES = [
    "jana gana mana adhinayaka jaya he bharata bhagya vidhata",
    "punjab sindhu gujarata maratha dravida utkala vanga",
    "hello world tokenizer benchmark deterministic",
] * 80

TESTS = {
    "case_1_oov_word": "xylophonically",
    "case_2_script_mismatch": "जन गण मन",
    "case_3_unseen_symbol": "hello 😀 world",
}


def run_model(name: str, config, corpus_path: str) -> None:
    tokenizer = Tokenizer.from_config(config)
    tokenizer.train([corpus_path], config)

    print(f"\n=== {name.upper()} ===")
    print(f"vocab_size={tokenizer.get_vocab_size()}")

    for label, text in TESTS.items():
        enc = tokenizer.encode(text)
        n_unk = enc.ids.count(0)
        print(f"\n{label}: {text!r}")
        print(f"tokens={enc.tokens}")
        print(f"ids={enc.ids}")
        print(f"unk_count={n_unk}, unk_rate={unk_rate([enc]):.3f}")


def main() -> None:
    configs = {
        "wordlevel": wordlevel_multilingual(vocab_size=200),
        "bpe": bpe_multilingual(vocab_size=200),
        "unigram": unigram_multilingual(vocab_size=200),
    }

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "task6_corpus.txt"
        corpus_path.write_text("\n".join(CORPUS_LINES), encoding="utf-8")

        for name, config in configs.items():
            run_model(name, config, str(corpus_path))


if __name__ == "__main__":
    main()
