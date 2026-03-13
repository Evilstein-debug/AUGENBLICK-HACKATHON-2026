import sys
import tempfile
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual, unigram_multilingual

CORPUS_LINES = ["Hello world"] * 10

def test_edge_cases():
    config = bpe_multilingual(vocab_size=100)
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "corpus.txt"
        corpus_path.write_text("\n".join(CORPUS_LINES), encoding="utf-8")

        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_path)], config)

        edge_cases = {
            "empty": "",
            "whitespace": "     \n   ",
            "mixed": "English   हिंदी",
            "emoji": "hello 🤔 world",
            "punctuation": "!!!???...",
            "long": "A" * 50000,
        }

        for name, text in edge_cases.items():
            print(f"\n--- Edge Case: {name} ---")
            try:
                encoding = tokenizer.encode(text)
                print(f"Len Tokens: {len(encoding.tokens)}")
                print(f"Offsets: {encoding.offsets[:10]}")
            except Exception as e:
                import traceback
                print(f"ERROR: {str(e)}")
                traceback.print_exc()

if __name__ == "__main__":
    test_edge_cases()
