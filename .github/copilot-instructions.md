# Copilot Instructions for this repository

Purpose: help AI coding assistants become productive quickly by documenting repository structure, workflows, and project-specific conventions.

## Big picture
- Goal: experimental Word2Vec / word-embedding implementations in NumPy and PyTorch.
- Major components (src/):
  - `content/`: dataset handling and local HuggingFace cache (see `src/content/mc4.py`).
  - `text_utils/`: tokenisation, vocabulary creation and helper transforms used across demos and tests.
  - `word2vec/` and `word2vec_negativesampling/`: NumPy reference implementations.
  - `word2vec_pytorch/`: PyTorch implementation and runnable demo (`word2vec_pytorch.py`).
  - `embedding_utils/`: analysis helpers (similarity, analogy helpers, NumPy + Torch variants).
  - `comparaisons/`: evaluation, metrics and plotting helpers.

## Run / developer commands (quick)
- Install deps: `pip install -r requirements.txt`
- Run tests: `pytest -q` (from repository root)
- Quick demo (PyTorch): `python src/word2vec_pytorch/word2vec_pytorch.py`
- Debugging: use the VS Code launch configuration `Python Debugger: Current File` (see `.vscode/launch.json`)

## Data & cache
- `src/content/mc4.py` configures `HF_HOME` and `HF_DATASETS_CACHE` to `src/content/cache`. HuggingFace datasets download there — these files are large; avoid committing `src/content/cache`.

## Project-specific conventions and patterns
- Backend suffixes: numeric/backend variants are explicit — look for `_numpy` vs `_torch` suffixes (e.g. in `src/embedding_utils/embedding_utils.py`).
- Files that act as demos/entrypoints: `src/word2vec_pytorch/word2vec_pytorch.py` and the NumPy scripts under `src/word2vec/`.
- Embedding tables: implementations keep two embedding matrices (`in_embed` and `out_embed`) — see `SkipGram` in `src/word2vec_pytorch/word2vec_pytorch.py`.
- Training helpers: `train_softmax()` and `train_neg()` exist in the PyTorch demo — change those for experimental training loop tweaks.
- Default tensor dtype: the PyTorch demo sets `torch.set_default_dtype(torch.float64)` — be mindful of dtype differences when comparing to NumPy.

## Where to change what
- Dataset / caching changes → `src/content/*` (keep cache local).
- Tokenisation / vocab → `src/text_utils/text_utils.py` and tests under `tests/text_utils/unit/test_text_utils.py`.
- NumPy algorithm changes → `src/word2vec/` and `src/word2vec_negativesampling/`.
- PyTorch experiments → `src/word2vec_pytorch/` (model, training functions, datasets classes: `SkipGramSoftmaxDS`, `SkipGramNegSampleDS`).
- Analysis and evaluation → `src/embedding_utils/` and `src/comparaisons/`.

## Testing conventions and examples
- Tests mirror module layout under `tests/` (e.g. `tests/word2vec/unit/test_matrice.py`).
- When adding features, add a unit test with fixed seeds for deterministic behavior.

## Integration points & notable functions
- Tokenisation and vocab: `tokenisation()`, `frequence_mots()`, `vocabulaire()` in `src/text_utils/text_utils.py`.
- Pair creation for skip-gram: `creation_paires()` (used to produce training pairs).
- PyTorch demo utilities: `mots_similaires_torch()` and `most_similar_to_vector_torch()` are in `src/embedding_utils/embedding_utils.py` and used by the demo.

## External dependencies & environment
- Key libs: `numpy`, `torch`, `datasets` (see `requirements.txt`).
- GPU-aware code: check `torch.cuda.is_available()` in PyTorch demo.
- Recommend using a virtual environment or conda; keep HF cache local to avoid large commits.

## Safety & PR guidance
- Do not commit large dataset/cache files under `src/content/cache`.
- Preserve naming conventions (`_numpy` vs `_torch`) and public APIs when editing existing modules.
- For algorithmic changes, add/modify a small unit test and run `pytest -q` locally before pushing.

## Quick references (examples)
- Run PyTorch demo: `python src/word2vec_pytorch/word2vec_pytorch.py`
- Run tests: `pytest -q`
- Check tokenisation: inspect `src/text_utils/text_utils.py` and `tests/text_utils/unit/test_text_utils.py`

If you'd like, I can tighten this further (e.g., add file links to specific functions or a short contributor checklist). Tell me what to expand.
