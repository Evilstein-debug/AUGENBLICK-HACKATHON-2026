import tempfile
from pathlib import Path

from abctokz.config.defaults import bpe_multilingual
from abctokz.config.schemas import BenchmarkConfig
from abctokz.eval.benchmark import BenchmarkRunner
from abctokz.tokenizer import Tokenizer

def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        corpus = tmpdir / "bench_corpus.txt"
        
        lines = ["hello world", "the quick brown fox", "नमस्ते दुनिया", "मराठी भाषेत"] * 500
        corpus.write_text("\n".join(lines), encoding="utf-8")
        
        tok_dir = tmpdir / "bpe_tok"
        cfg = bpe_multilingual(vocab_size=200)
        tok = Tokenizer.from_config(cfg)
        tok.train([str(corpus)], cfg)
        tok.save(str(tok_dir))
        
        bench_cfg = BenchmarkConfig(
            name="trust_test",
            corpus_paths=[str(corpus)],
            tokenizer_paths=[str(tok_dir)],
            sample_size=2000,
            warmup_runs=3,
            timed_runs=10,
            output_dir=str(tmpdir),
            languages=["mixed"]
        )
        
        runner = BenchmarkRunner(bench_cfg)
        
        print("--- EXECUTION RUN 1 ---")
        res1 = runner.run()[0]
        
        print("\n--- EXECUTION RUN 2 ---")
        res2 = runner.run()[0]
        
        print("\n--- COMPARISON ---")
        print(f"Fertility (Run 1 vs 2): {res1.fertility} vs {res2.fertility}")
        print(f"UNK Rate  (Run 1 vs 2): {res1.unk_rate} vs {res2.unk_rate}")
        print(f"Throughput(Run 1 vs 2): {res1.throughput_sps:.2f} sps vs {res2.throughput_sps:.2f} sps")

if __name__ == "__main__":
    main()