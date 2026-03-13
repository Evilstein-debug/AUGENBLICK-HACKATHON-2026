import sys
import tempfile
from pathlib import Path

# Add the 'src' directory to Python's module search path so we can import abctokz
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual, unigram_multilingual
from abctokz.eval.metrics import fertility

# A small inline corpus combining English and Devanagari 
# to ensure both models can build up their vocabularies.
CORPUS_LINES = [
    "This is an example English sentence.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming technology.",
    "Tokenization is the first step in NLP.",
    "आयो लाल, सभई चायो, झूलेलाल!",
    "गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!",
    "नमस्ते दुनिया",
    "भारत एक महान देश है।",
    "मराठी आणि हिंदी या देवनागरी लिपी वापरतात.",
] * 200 # Multiple copies to simulate a realistic minimum corpus size for WordLevel/BPE

def run_fertility(phrase, model_type, vocab_size):
    # 1. Pick the corresponding config generator
    if model_type == "bpe":
        config = bpe_multilingual(vocab_size=vocab_size)
    else:  # unigram
        config = unigram_multilingual(vocab_size=vocab_size)
    
    # 2. Build and train on a temporary corpus
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "corpus.txt"
        corpus_path.write_text("\n".join(CORPUS_LINES), encoding="utf-8")

        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_path)], config)
        
        # 3. Encode the phrase
        encoding = tokenizer.encode(phrase)
        
        # 4. Calculate fertility
        # Fertility is defined as: len(tokens) / len(reference_words)
        ref_word_count = len(phrase.split())
        score = fertility([encoding], [ref_word_count])
        
        return score, encoding.tokens

if __name__ == "__main__":
    phrases = {
        "Sindhi": "आयो लाल, सभई चायो, झूलेलाल!",
        "Marathi": "गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!"
    }

    vocab_sizes = [100, 400, 800]
    models = ["bpe", "unigram"]

    print("=== Fertility Scores ===")
    for name, text in phrases.items():
        print(f"\n--- Phrase: {name} ---")
        for model in models:
            for vs in vocab_sizes:
                try:
                    score, tokens = run_fertility(text, model, vs)
                    print(f"[{model.upper()} | Vocab: {vs:<3}] Fertility: {score:.2f} | Tokens: {tokens}")
                except Exception as e:
                    print(f"[{model.upper()} | Vocab: {vs:<3}] Error: {str(e)}")
