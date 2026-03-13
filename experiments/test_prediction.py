import tempfile
from pathlib import Path

from abctokz.config.defaults import bpe_multilingual
from abctokz.config.schemas import TokenizerConfig
from abctokz.tokenizer import Tokenizer
from abctokz.eval.metrics import fertility, round_trip_success_rate

def main():
    test_sentences = [
        "Hello, world!",
        "Wait, what?",
        "Tokenization is hard.",
        "नमस्ते, दुनिया!"
    ]
    
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        corpus = tmpdir / "corpus.txt"
        
        # Build a tiny corpus with lots of punctuation
        lines = ["Hello, world!", "Wait, what?", "Tokenization is hard.", "नमस्ते, दुनिया!"] * 50
        corpus.write_text("\n".join(lines), encoding="utf-8")
        
        # --- RUN 1: THE DEFAULT CONFIG (Baseline) ---
        default_cfg = bpe_multilingual(vocab_size=100)
        tok_default = Tokenizer.from_config(default_cfg)
        tok_default.train([str(corpus)], default_cfg)
        
        enc_default = tok_default.encode_batch(test_sentences)
        dec_default = [tok_default.decode(e.ids) for e in enc_default]
        fert_default = fertility(enc_default, [len(s.split()) for s in test_sentences])
        rt_default = round_trip_success_rate(test_sentences, dec_default)
        
        mutated_dict = default_cfg.model_dump()
        
        # We override the sequence pre-tokenizer to ONLY use whitespace
        mutated_dict["pretokenizer"] = {"type": "whitespace"} 
        
        mutated_cfg = TokenizerConfig.model_validate(mutated_dict)
        
        tok_mutated = Tokenizer.from_config(mutated_cfg)
        tok_mutated.train([str(corpus)], mutated_cfg)
        
        enc_mutated = tok_mutated.encode_batch(test_sentences)
        dec_mutated = [tok_mutated.decode(e.ids) for e in enc_mutated]
        fert_mutated = fertility(enc_mutated, [len(s.split()) for s in test_sentences])
        rt_mutated = round_trip_success_rate(test_sentences, dec_mutated)

        print("--- THE PREDICTION TEST ---")
        print("\n[Baseline: Default Config]")
        print(f"Fertility: {fert_default:.2f}")
        print(f"Round Trip Success: {rt_default * 100:.1f}%")
        print("Tokens for 'Hello, world!':", enc_default[0].tokens)
        
        print("\n[Mutated: Punctuation Pre-Tokenizer Removed]")
        print(f"Fertility: {fert_mutated:.2f}")
        print(f"Round Trip Success: {rt_mutated * 100:.1f}%")
        print("Tokens for 'Hello, world!':", enc_mutated[0].tokens)

if __name__ == "__main__":
    main()