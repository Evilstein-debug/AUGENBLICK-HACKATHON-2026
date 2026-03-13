from __future__ import annotations

from pathlib import Path
import tempfile
import unicodedata
import shutil

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual
from abctokz.normalizers import build_normalizer
from abctokz.eval.metrics import round_trip_success_rate


def train_tokenizer(corpus_path: str, out_dir: str, vocab_size: int = 500) -> Tokenizer:
    cfg = bpe_multilingual(vocab_size=vocab_size)
    tokenizer = Tokenizer.from_config(cfg)
    tokenizer.train([corpus_path], cfg)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir)
    return Tokenizer.load(out_dir)


def run_round_trip_tests(tokenizer: Tokenizer, samples: list[str], normalizer=None) -> None:
    decoded = []
    for s in samples:
        enc = tokenizer.encode(s)
        out = tokenizer.decode(enc.ids)
        decoded.append(out)
        print("\nInput repr:", repr(s))
        print("Decoded repr:", repr(out))
        if s == out:
            print("  -> Exact round-trip: YES")
        else:
            print("  -> Exact round-trip: NO")
            if normalizer is not None:
                norm = normalizer.normalize(s)
                print("  Normalized input repr:", repr(norm))
                if norm == out:
                    print("  -> Decoded matches normalized input (normalization explains difference)")
    targets = None
    if normalizer is not None:
        targets = [normalizer.normalize(s) for s in samples]
    rate = round_trip_success_rate(samples, decoded, normalized_originals=targets)
    print(f"\nRound-trip success rate (using normalized targets if provided): {rate:.3f}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        corpus = tmpdir / "corpus.txt"
        corpus_lines = [
            "hello world",
            "tokenization is important",
            "नमस्ते दुनिया",
            "भारत",
            "hello नमस्ते",
        ] * 50
        corpus.write_text("\n".join(corpus_lines), encoding="utf-8")

        outdir = Path("artifacts/round_trip_bpe")
        if outdir.exists():
            shutil.rmtree(outdir)

        print("Training BPE tokenizer for round-trip tests...")
        tokenizer = train_tokenizer(str(corpus), str(outdir), vocab_size=500)

        cfg = bpe_multilingual(vocab_size=500)
        normalizer = build_normalizer(cfg.normalizer) if cfg.normalizer else None

        samples = [
            "hello world",
            "नमस्ते दुनिया",
            "hello नमस्ते world",
            unicodedata.normalize("NFC", "नमस्ते"),
            unicodedata.normalize("NFD", "नमस्ते"),
            unicodedata.normalize("NFC", "résumé"),
            unicodedata.normalize("NFD", "résumé"),
        ]

        print("\nRunning round-trip tests on samples (including NFC vs NFD)...")
        run_round_trip_tests(tokenizer, samples, normalizer=normalizer)


if __name__ == "__main__":
    main()
