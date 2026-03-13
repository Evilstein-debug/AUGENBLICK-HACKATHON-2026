"""Task 12: Demonstrate decode() intent vs reality gap."""
from __future__ import annotations

import tempfile
from pathlib import Path

from abctokz import Tokenizer
from abctokz.config.defaults import wordlevel_multilingual
from abctokz.eval.metrics import round_trip_success_rate

CORPUS_LINES = [
    "hello <div> world",
    "hello <div> world",
    "hello world",
] * 40

TEST_TEXT = "hello <div> world"


def main() -> None:
    config = wordlevel_multilingual(vocab_size=100)

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "task12_corpus.txt"
        corpus_path.write_text("\n".join(CORPUS_LINES), encoding="utf-8")

        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_path)], config)

        enc = tokenizer.encode(TEST_TEXT)

        decoded_default = tokenizer.decode(enc.ids)  # skip_special_tokens=True (default)
        decoded_keep_all = tokenizer.decode(enc.ids, skip_special_tokens=False)

        print("=== Task 12 decode() gap demo ===")
        print(f"special_tokens_defined={list(tokenizer._special_tokens.keys())}")
        print(f"input={TEST_TEXT!r}")
        print(f"encoded_tokens={enc.tokens}")
        print(f"encoded_ids={enc.ids}")
        print(f"decoded_default={decoded_default!r}")
        print(f"decoded_skip_special_false={decoded_keep_all!r}")
        print(
            "round_trip_success_rate(default)=",
            f"{round_trip_success_rate([TEST_TEXT], [decoded_default]):.3f}",
        )


if __name__ == "__main__":
    main()
