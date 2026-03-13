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


### Task 2 — Who Does What? Mapping Module Responsibilities

The architecture is actually pretty standard for a modern Python library—everything is strictly separated, with a few weird exceptions. Here is how the responsibilities map out.

**1. Module Responsibility Mapping**

* **Training a tokenizer (learning vocabulary from text):** All the heavy lifting for crunching a corpus and building a vocabulary happens in `src/abctokz/trainers/`. If you look inside, you'll see specific scripts like bpe_trainer.py and unigram_trainer.py that handle the actual statistical learning.

* **Using a trained tokenizer to encode new text:** This is a two-part job. The main one is `src/abctokz/tokenizer.py` (the core orchestrator). It takes the string and routes it through the pipeline, but the actual math and subword extraction is outsourced to the engines living inside `src/abctokz/models/` (which holds the logic for BPE, Unigram, and WordLevel).

* **Saving and loading a tokenizer to/from disk:** Surprisingly, this isn't given to a dedicated storage service. It is implemented directly inside `src/abctokz/tokenizer.py` using the save() and load() methods on the main tokenizer class, which personally is not what I admire.

* **Measuring tokenizer quality (fertility, UNK rate, etc.):** Everything related to grading the model lives in the `src/abctokz/eval/` directory. Specifically, metrics.py handles scores (like counting unknown tokens), while benchmark.py is dedicated to measuring system throughput and latency.

* **Comparing abctokz against external tokenizers:** This is heavily isolated inside `src/abctokz/adapters/`, which contains dedicated wrapper scripts like hf.py (for Hugging Face) and sentencepiece.py (for Google's SentencePiece).

**2. A Clean Module Boundary**

What I really loved finding in this codebase was how they handled external dependencies—specifically the **`src/abctokz/adapters/`** module.

**Why it is satisfyingly clean:** External machine learning libraries like HuggingFace’s `transformers` or Google’s `sentencepiece` are absolutely large. They carry massive, heavy dependencies. The architects of this repo were smart enough to lock all of that code entirely inside the `adapters/` folder. 

If we look at the import trees, the core `tokenizer.py` orchestrator and the `models/` directory *never* import anything from `adapters/`. It acts as a strict Anti-Corruption Layer. In practice, this means if a standard user just wants to run this tokenizer locally, they aren't forced to download gigabytes of third-party ML libraries just to get the code to compile, which is what I liked very much. 

**3. A Blurry or Inconsistent Module Boundary**

On the flip side, the boundary feels highly inconsistent and blurry within **`src/abctokz/tokenizer.py`**—specifically when you look at how the `save()` and `load()` methods are written.

**Why it feels blurry:** The `AugenblickTokenizer` class is clearly designed to be a high-level text processing orchestrator. Its whole job is to route strings through Normalization → Pre-tokenization → Model. 
However, when you scroll down to its `save()` method, it suddenly starts doing low-level file system operations. It's actively creating directories (`mkdir`), manually calculating SHA256 checksums, and building raw JSON manifests from scratch. It violently mixes core text-transformation logic with direct hard-drive I/O operations, which is a classic violation of the Single Responsibility Principle. It makes the orchestrator a "Fat Controller."

```
def save(self, path: str) -> None:
        out = ensure_dir(path)
        self._model.save(str(out))

        st_data = {k: v.to_dict() for k, v in self._special_tokens.items()}
        save_json(st_data, out / SPECIAL_TOKENS_FILENAME)

        model_type = self._infer_model_type()
        config_data: dict[str, object] = {"model_type": model_type, "schema_version": SCHEMA_VERSION}
        save_json(config_data, out / CONFIG_FILENAME)

        vocab_size = self._model.get_vocab_size()
        checksum = ""
        vocab_path = out / "vocab.json"
        if vocab_path.exists():
            checksum = sha256_file(vocab_path)

        metadata = ArtifactMetadata(
            schema_version=SCHEMA_VERSION,
            model_type=model_type,
            vocab_size=vocab_size,
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
            checksum=checksum,
        )
        save_json(metadata.to_dict(), out / MANIFEST_FILENAME)
        logger.info("Tokenizer saved to %s (vocab_size=%d)", path, vocab_size)
```

**What I would do about it:** I wouldn't tear up the repo to create a brand new architecture. I would just respect the current setup and move this messy saving logic into the already-existing `src/abctokz/vocab/serialization.py` file, or just append it to the helper functions in `src/abctokz/utils/io.py`.

The `AugenblickTokenizer` class shouldn't have to care about how or where its files are saved to a hard drive; it should just blindly pass its state to an external save function and let the I/O module handle the JSON dumping.

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

### TASK 4 — How Does a Config Become a Tokenizer?



### Task 5 — Is It Truly Deterministic?

The documentation claims the training output is perfectly repeatable, but I rarely trust docs until I've verified it myself. I set up a quick experiment (`experiments/determinism_bpe.py`) to train the `abctokz` BPE model twice from scratch on the exact same dataset to see if anything drifted.

### What IS Deterministic?

So, the experiment found that the core artifacts are bulletproof. Our tests confirmed these outputs are **100% identical** across runs:

* **Vocabulary Order (`vocab.json`):** Both generated files were byte-for-byte matches (exactly 2529 chars). If you dig into the training code, it explicitly sorts frequency ties alphabetically. That prevents the random hash-map ordering bugs that usually plague Python dictionaries.
* **Merge Rules (`merges.txt`):** The exact sequence of BPE merges is identical every single time (2231 chars). 
* **Token Outputs:** Encoding the same benchmark text through both tokenizer instances fired back the exact same token arrays and ID numbers.

*(I've attached a screenshot of the terminal output showing the matching arrays and file diffs:*

![OUTPUT_SCREENSHOT](/public/images/experiment_bpe.png)

### What is NOT Deterministic

While the compiled models are identical, the hardware execution is not:

* **Training Time:** Run 1 clocked in at 0.0913 seconds, while Run 2 immediately dropped to 0.0099 seconds. 

This isn't a bug in the code. It is just the reality of CPU caching (the data was already warm in memory for the second run), thermal throttling, and OS background noise. The time varies, but it doesn't lessen the final tokenizer artifacts.

### Remaining Risks

Even though this specific local test passed, determinism is incredibly fragile. Here are the edge cases that I think could realistically break this codebase:

1. **Cross-Platform Math:** The Unigram model relies heavily on floating-point probabilities (`math.log`). Different CPU architectures (like an M-series Mac vs. an Intel server) handle deep decimal rounding slightly differently. A tiny precision difference at the 15th decimal place can flip a probability tie-breaker, completely altering the token boundaries.
2. **Future Multi-threading:** Right now, the data pipeline is single-threaded. If a developer later tries to optimize the `_corpus_iter()` generator by parallelizing the file reads without strictly enforcing the order of the results, chunks of text will hit the trainer in a random sequence. That race condition will immediately destroy the determinism..

```
def _corpus_iter():
            for path in corpus_paths:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        if normalizer:
                            line = normalizer.normalize(line)
                        if pretokenizer:
                            line = " ".join(pretokenizer.pre_tokenize(line))
                        yield line
```