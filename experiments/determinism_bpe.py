# experiments/determinism_bpe.py
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual


def train_and_save(corpus_path: str, out_dir: str, vocab_size: int = 500) -> float:
    """Trains the tokenizer and returns the time taken in seconds."""
    cfg = bpe_multilingual(vocab_size=vocab_size)
    tokenizer = Tokenizer.from_config(cfg)
    
    start_time = time.perf_counter()
    tokenizer.train([corpus_path], cfg)
    end_time = time.perf_counter()
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir)
    
    return end_time - start_time


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = tmp / "corpus.txt"
        lines = [
            "hello world",
            "the quick brown fox jumps over the lazy dog",
            "tokenization is important for natural language processing",
            "नमस्ते दुनिया",
            "भारत एक विशाल देश है",
            "hello नमस्ते world दुनिया",
        ] * 50
        corpus.write_text("\n".join(lines), encoding="utf-8")

        base = Path("artifacts/determinism_test")
        if base.exists():
            shutil.rmtree(base)

        run1 = base / "run1"
        run2 = base / "run2"

        print("--- Training Phase ---")
        time_run1 = train_and_save(str(corpus), str(run1), vocab_size=500)
        print(f"Run 1 completed in {time_run1:.4f} seconds.")
        
        time_run2 = train_and_save(str(corpus), str(run2), vocab_size=500)
        print(f"Run 2 completed in {time_run2:.4f} seconds.")
        print(f"Time difference: {abs(time_run1 - time_run2):.4f} seconds (Expected: Non-deterministic)")

        # Compare vocab and merges
        v1 = read_text(run1 / "vocab.json")
        v2 = read_text(run2 / "vocab.json")
        m1 = read_text(run1 / "merges.txt")
        m2 = read_text(run2 / "merges.txt")

        print('\n--- Artifact Comparison ---')
        print(f"vocab.json identical: {v1 == v2} (Length: {len(v1)} chars)")
        print(f"merges.txt identical: {m1 == m2} (Length: {len(m1)} chars)")

        # Load tokenizers and compare encodings
        t1 = Tokenizer.load(str(run1))
        t2 = Tokenizer.load(str(run2))

        samples = [
            "hello world",
            "नमस्ते दुनिया",
            "hello नमस्ते world",
            "tokenization",
        ]

        print('\n--- Encoding Comparison ---')
        all_match = True
        for s in samples:
            e1 = t1.encode(s)
            e2 = t2.encode(s)
            
            ids_match = e1.ids == e2.ids
            toks_match = e1.tokens == e2.tokens
            
            print(f"\nSample: '{s}'")
            print(f"  Run 1 IDs:     {e1.ids}")
            print(f"  Run 2 IDs:     {e2.ids}")
            print(f"  IDs Match:     {ids_match}")
            print(f"  Run 1 Tokens:  {e1.tokens}")
            print(f"  Run 2 Tokens:  {e2.tokens}")
            print(f"  Tokens Match:  {toks_match}")
            
            if not (ids_match and toks_match):
                all_match = False

        print('\n--- Final Verdict ---')
        print(f"Overall encoding identical: {all_match}")


if __name__ == '__main__':
    main()