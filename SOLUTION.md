# Solutions to Tasks

# Solutions to Tasks

### TASK 1

### Question :

Take this Sanskrit mantra:

ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥

Pick one model (BPE is a good choice), train it on a small corpus, and encode this string. Then trace what actually
happened — from the moment you called encode() all the way to the final list of token IDs.

What we're looking for:

    Which files and classes were involved at each stage?
    What did the normalizer do to the string before the model saw it?
    What did the pre-tokenizer do after normalization?
    How did the model turn pre-tokens into subword pieces?
    How were those pieces turned back into a string during decode()?

### ANSWER:

***background:***

spent HOURS understanding how llms work, stages of tokenization and general knowledge to even read this project's README

***Understanding the context:***

Very generously, within the examples folder we have ready to run code samples including that of bpe,

THE CORPUS IS GIVEN
observation about the corpus: for the given text to encode with is in sanskrit, the corpus has only few
lines of sanskrit text.

so umm yeah wild ride for the text to get encoded.

ill paste the output first, and then try my best to explain what happened

![OUTPUT_SCREENSHOT](reference_images/task1.png)

***Files modified for the output:***

added the input line within examples/train_bpe.py files
line 65 to be precise

Explanation:

straight of the bat, we can see the input line is itself messed up, this is a pycharm, vscode terminal issue where the
support for foreign languages arent supported.

1. *Which files and classes were involved at each stage?*

the train_bpe.py and  encode() & decode() function is written in tokenizer.py
additionally a temporary ghost "corpus" is created for the script to run containing all the corpus*30 and then deleted
automatically after the script is run (pretty nice automation that)

the code block for encoder function:

(line 93)

```aiignore
 def encode(self, text: str) -> Encoding:
        """Encode *text* into an :class:`~abctokz.types.Encoding`.

        The full pipeline is applied: normalization → pre-tokenization
        → model tokenization → post-processing.

        Args:
            text: Raw input string.

        Returns:
            :class:`~abctokz.types.Encoding` with ids, tokens, and offsets.
        """
``` 

2. *What did the normalizer do to the string before the model saw it?*

in this specific line of encoder function implemention

```aiignore
        normalized = self._normalizer.normalize(text) if self._normalizer else text
```

It checks if a normalizer is attached to this tokenizer instance. If so, it passes the raw text through it (e.g.,
lowercasing it). If not, it just passes the raw text through untouched

if a normalizer exists, te rest of function operates on the variable *normalized* and not the original text

3. *What did the pre-tokenizer do after normalization?*

in the specific line of encoder function

```aiignore
        if self._pretokenizer:
            pre_tokens = self._pretokenizer.pre_tokenize(normalized)
        else:
            pre_tokens = [normalized]
```

does really the same thing as the normalizer block, checks if a instance of pretokenizer is attacked, if not you can see
it just directly puts the normalized text in a list, untouched

else it pre tokenizes it

### OPTIMIZATION SUGGESTION

*maybe if we can run a custom pretokenize script instead of skipping the pretokenization process if there is no custom
pre tokenizer?*

4. How did the model turn pre-tokens into subword pieces?

it does this in the 3rd stage, the tokenization process

```aiignore
        cursor = 0
        for pre_tok in pre_tokens:
            # Find the offset of this pre_token in the normalized string
            start_pos = normalized.find(pre_tok, cursor)
            if start_pos == -1:
                start_pos = cursor

            pairs = self._model.tokenize(pre_tok)
            char_offset = start_pos
            for tok_str, tok_id in pairs:
                tok_len = len(tok_str.lstrip("##"))  # strip continuation prefix for offset
                ids.append(tok_id)
                tokens.append(tok_str)
                offsets.append((char_offset, char_offset + len(pre_tok)))
                is_special = int(tok_str in self._special_tokens)
                special_mask.append(is_special)
                attention_mask.append(1)

            cursor = start_pos + len(pre_tok)
```

here it iterates through every sub token in the pre_token list, and it runs the bpe algorithm for every word,
appending "##" and its unique id for every token.

5. How were those pieces turned back into a string during decode()?

yet to do



## Task 2 — Who Does What? Mapping Module Responsibilities

**1. Module Responsibility Mapping**
* **Training a tokenizer (learning vocabulary from text):** Handled by `src/abctokz/trainers/` (e.g., bpe_trainer.py, unigram_trainer.py).

* **Using a trained tokenizer to encode new text:** Orchestrated by `src/abctokz/tokenizer.py` (the main API controller), which routes the text into the core logic engines located in `src/abctokz/models/` (the subword extraction algorithms like BPE, Unigram and Wordlevel).

* **Saving and loading a tokenizer to/from disk:** Implemented directly within `src/abctokz/tokenizer.py` via the `save()` and `load()` class methods on `src/abctokz/models/`.

* **Measuring tokenizer quality (fertility, UNK rate, etc.):** Located in `src/abctokz/eval/metrics.py` and `src/abctokz/eval/benchmark.py`.

* **Comparing abctokz against external tokenizers:** Isolated inside `src/abctokz/adapters/` (specifically `hf.py` (Hugging Face) and `sentencepiece.py` (Sentence Piece)).

**2. A Clean Module Boundary**
The most satisfying architectural boundary in this codebase is the **`src/abctokz/adapters/`** module. 

**Why it is clean:** External machine learning libraries like HuggingFace's `transformers` or Google's `sentencepiece` are massive, heavy dependencies. The repo isolated these inside the `adapters/` folder. 

The core `tokenizer.py` orchestrator and the `models/` directory never import from `adapters/`. This acts as a strict Anti-Corruption Layer. It ensures that standard users can run the core tokenizer without being forced to download gigabytes of third-party libraries.

**3. A Blurry or Inconsistent Module Boundary**
The boundary feels highly blurry within **`src/abctokz/tokenizer.py`**, specifically regarding the `save()` and `load()` methods.

**Why it is blurry:** The `AugenblickTokenizer` class is supposed to coordinate the text processing pipeline (Normalization → Pre-tokenization → Model). However, its `save()` method suddenly takes on low-level file system tasks—like creating directories, calculating checksums, and writing JSON files. This mixes core tokenization logic with direct file storage operations, which violates the Single Responsibility Principle.

**What I would do about it:**
Instead of creating brand new folders, I would stick to the current setup and move this saving logic into the existing `src/abctokz/vocab/serialization.py` file, or add it to the helpers in `src/abctokz/utils/io.py`. The AugenblickTokenizer shouldn't have to care about how files are saved to the hard drive. It should just pass the data to an external save function instead of building the JSON files manually.

---

## Task 5 — Is It Truly Deterministic?

When training tokenizers, the output must be perfectly repeatable. We ran an experiment (`experiments/determinism_bpe.py`) to prove that training `abctokz` twice on the exact same data produces identical results. 

### What IS Deterministic?

Our test confirmed the following outputs are **100% identical** across runs:

* **Vocabulary Order (`vocab.json`):** Both files were byte-for-byte matches (2529 chars). The code explicitly sorts ties alphabetically, preventing random ordering.
* **Merge Rules (`merges.txt`):** The exact sequence of BPE merges is identical every time (2231 chars).
* **Token Outputs:** Encoding the same text through both tokenizers produced the exact same token strings and ID numbers.

*(See the [attached screenshot](/public//images/experiment_bpe.png) for the terminal output showing matching arrays and exact file comparisons).*

### What is NOT Deterministic

While the final model files are identical, the training process varies:

* **Training Time:** As seen in our logs, Run 1 took 0.0913 seconds, while Run 2 took 0.0099 seconds. 

This happens because of the operating system's background tasks, CPU caching, and thermal throttling. It is **completely normal** and doesn't affect the final tokenizer artifacts.

### Remaining Risks

Even with solid code, determinism can fail under these specific edge cases:

1. **Different Computers (CPU/OS):** Unigram models use decimal math (`math.log`). Different processors (like an ARM Mac vs. an Intel PC) might round very long decimals differently, which could change tie-breaker results.
2. **Future Multi-threading:** If a developer updates the code to load data in parallel (multiple threads at once) without strictly ordering the results, the training data will mix randomly and break determinism.

### Task 3 — The National Anthem Test

**Approach**

The experiment uses the existing `abctokz` BPE pipeline, no new code was written. The full script is at [`examples/task3_national_anthem.py`](examples/task3_national_anthem.py).

**Step-by-step**

1. Build a small training corpus containing both English transliteration and Devanagari lines of Jana Gana Mana (first stanza), plus a few supplementary Hindi/Marathi lines. Each line is repeated 50× so word frequencies exceed the default `min_frequency=2` threshold in `BPETrainerConfig`.
2. Create a `bpe_multilingual(vocab_size=500)` config defined in `src/abctokz/config/defaults.py`. This wires together:
	 - `multilingual_shared_normalizer()` (NFC Unicode normalization + whitespace collapse)
	 - `DevanagariAwarePreTokenizer` (splits on script boundaries, not just spaces)
	 - `BPEModel` + `BPETrainer`
3. Instantiate `Tokenizer.from_config(config)` and call `tokenizer.train([corpus_path], config)`.
4. Encode the two test sentences and call `fertility()` from `src/abctokz/eval/metrics.py`.
5. Pass the same strings through `tiktoken.encoding_for_model("gpt-4")` and compare.

**Test sentences**

| | Text |
|---|---|
| English transliteration | `Jana Gana Mana Adhinayaka Jaya He Bharata Bhagya Vidhata Punjab Sindhu` |
| Devanagari | `जन गण मन अधिनायक जय हे भारत भाग्य विधाता पंजाब सिंधु` |

Both sentences have **11 whitespace-separated words**.

**Results — abctokz BPE (vocab size trained: 333 / requested: 500)**

```
── English transliteration ──
	Input : Jana Gana Mana Adhinayaka Jaya He Bharata Bhagya Vidhata Punjab Sindhu
	Tokens (38): ['J', '##an', '##a', 'G', '##an', '##a', 'M', '##an', '##a','A', '##dh', '##in', '##a', '##ya', '##ka', 'Ja', '##ya', 'He','B', '##ha', '##ra', '##ta', 'B', '##ha', '##g', '##ya','V', '##id', '##ha', '##ta', 'P', '##un', '##ja', '##b','S', '##in', '##dh', '##u']

	IDs  : [228, 15, 1, 220, 15, 1, 236, 15, 1, 211, 30, 56, 1, 92, 63, 234, 92,226, 214, 44, 77, 82, 214, 44, 35, 92, 254, 54, 44, 82, 242, 88, 60,19, 244, 56, 30, 85]

── Devanagari ──
	Input : जन गण मन अधिनायक जय हे भारत भाग्य विधाता पंजाब सिंधु
	Tokens (26): ['जन', 'गण', 'मन', 'अ', '##धि', '##ना', '##य', '##क', 'जय','हे', 'भा', '##रत', 'भा', '##ग', '##्य', 'वि', '##धा', '##ता','प', '##ंज', '##ा', '##ब', 'स', '##ि', '##ंध', '##ु']
	IDs  : [281, 275, 312, 264, 145, 151, 165, 103, 282, 331, 308, 173, 308, 112,209, 322, 144, 134, 297, 101, 189, 154, 325, 194, 102, 197]
```

**Fertility scores**

| Script | Tokens | Words | Fertility (tokens ÷ words) |
|--------|--------|-------|---------------------------|
| English transliteration (abctokz) | 38 | 11 | **3.455** |
| Devanagari (abctokz) | 26 | 11 | **2.364** |
| English transliteration (GPT-4 / tiktoken) | 23 | 11 | **2.091** |
| Devanagari (GPT-4 / tiktoken) | 54 | 11 | **4.909** |

**Why the numbers differ**

*abctokz English > abctokz Devanagari (3.455 vs 2.364)*

The model was trained on a Devanagari-rich corpus, so the BPE merger rules learned longer Devanagari subword units (e.g. `जन`, `गण`, `मन` appear as single tokens). English transliteration, by contrast, appears in fewer training examples meaning fewer high-frequency character n-grams were merged, leaving many single-character pieces (`J`, `G`, `M`, …) with `##` continuations.

The `DevanagariAwarePreTokenizer` (see `src/abctokz/pretokenizers/devanagari_aware.py`) also plays a role: it splits on script boundaries, ensuring Devanagari characters are never mixed with Latin characters in a single pre-token, giving the BPE trainer clean Devanagari units to learn merge rules on.

*tiktoken English < tiktoken Devanagari (2.091 vs 4.909)*

This is the exact opposite pattern. GPT-4's tokenizer was trained on a corpus overwhelmingly dominated by Latin-script text, so it has large English vocabulary entries and merges English words efficiently. Devanagari, however, is low-resource in that training set — the tokenizer has almost no merged Devanagari pieces, so every Unicode code point (consonant, vowel sign, anusvara) becomes its own token, pushing fertility above 4.9.

This comparison makes the design intent of `abctokz` concrete: the library exists precisely to close this gap for Devanagari-script languages.

**Bonus — tiktoken comparison output**

```
── tiktoken (GPT-4) comparison ──
	English    : 23 tokens  (fertility 2.091)
	Devanagari : 54 tokens  (fertility 4.909)
```

GPT-4 is 1.6× more efficient on English but **2.1× less efficient on Devanagari** compared to an `abctokz` BPE tokenizer trained on a balanced bilingual corpus. This confirms that a purpose-built multilingual tokenizer is meaningful for Indic-script workloads.