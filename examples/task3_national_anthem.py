"""Task 3: The National Anthem Test"""
from __future__ import annotations
import tempfile
from pathlib import Path

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual
from abctokz.eval.metrics import fertility

CORPUS_LINES = [
    "Jana Gana Mana Adhinayaka Jaya He Bharata Bhagya Vidhata",
    "Punjab Sindhu Gujarata Maratha Dravida Utkala Vanga",
    "Vindhya Himachala Yamuna Ganga Uchchala Jaladhi Taranga",
    "Tava Shubha Name Jage Tava Shubha Ashisha Mange",
    "Gahe Tava Jayagatha",
    "Janata Bhagya Vidhata Jaya He Bharata Bhagya Vidhata",
    "जन गण मन अधिनायक जय हे भारत भाग्य विधाता",
    "पंजाब सिंधु गुजरात मराठा द्राविड़ उत्कल बंगा",
    "विंध्य हिमाचल यमुना गंगा उच्छल जलधि तरंगा",
    "तव शुभ नामे जागे तव शुभ आशीष मांगे",
    "गाहे तव जयगाथा",
    "जनता भाग्य विधाता जय हे भारत भाग्य विधाता",

    "hello world नमस्ते दुनिया",
    "हिन्दी भाषा में परीक्षण",
    "मराठी भाषेत टोकनायझेशन",
    "भारत एक विशाल देश है",
] * 50

EN_TEXT = "Jana Gana Mana Adhinayaka Jaya He Bharata Bhagya Vidhata Punjab Sindhu"
HI_TEXT = "जन गण मन अधिनायक जय हे भारत भाग्य विधाता पंजाब सिंधु"

def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "corpus.txt"
        corpus_path.write_text("\n".join(CORPUS_LINES), encoding="utf-8")

        config = bpe_multilingual(vocab_size=500)
        tokenizer = Tokenizer.from_config(config)
        print("Training BPE tokenizer...")
        tokenizer.train([str(corpus_path)], config)
        print(f"  Vocab size: {tokenizer.get_vocab_size()}")

        enc_en = tokenizer.encode(EN_TEXT)
        enc_hi = tokenizer.encode(HI_TEXT)

        print(f"\n── English transliteration ──")
        print(f"  Input: {EN_TEXT}")
        print(f"  Tokens ({len(enc_en.ids)}): {enc_en.tokens}")
        print(f"  IDs:   {enc_en.ids}")

        print(f"\n── Devanagari ──")
        print(f"  Input: {HI_TEXT}")
        print(f"  Tokens ({len(enc_hi.ids)}): {enc_hi.tokens}")
        print(f"  IDs:   {enc_hi.ids}")

        en_words = len(EN_TEXT.split())
        hi_words = len(HI_TEXT.split())

        fert_en = fertility([enc_en], [en_words])
        fert_hi = fertility([enc_hi], [hi_words])

        print(f"\n── Fertility ──")
        print(f"  English : {len(enc_en.ids)} tokens / {en_words} words = {fert_en:.3f}")
        print(f"  Devanagari: {len(enc_hi.ids)} tokens / {hi_words} words = {fert_hi:.3f}")

        try:
            import tiktoken
            enc_gpt = tiktoken.encoding_for_model("gpt-4")
            gpt_en  = enc_gpt.encode(EN_TEXT)
            gpt_hi  = enc_gpt.encode(HI_TEXT)
            print(f"\n── tiktoken (GPT-4) comparison ──")
            print(f"  English    : {len(gpt_en)} tokens  (fertility {len(gpt_en)/en_words:.3f})")
            print(f"  Devanagari : {len(gpt_hi)} tokens  (fertility {len(gpt_hi)/hi_words:.3f})")
        except ImportError:
            print("\n[skip] tiktoken not installed — run: pip install tiktoken")

if __name__ == "__main__":
    main()