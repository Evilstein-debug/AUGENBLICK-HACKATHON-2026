# Solutions to Tasks


### Task 2 — Who Does What? Mapping Module Responsibilities

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