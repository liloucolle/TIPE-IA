from pathlib import Path
import os

# Resolve the folder containing *this* file; fall back to CWD (e.g., notebooks)
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Put all HF caches in a local subfolder next to your script
CACHE_DIR = SCRIPT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Make Hugging Face libs use this cache (works across platforms)
os.environ["HF_HOME"] = str(CACHE_DIR)                 # base for HF caches
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR)       # datasets cache
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "transformers")  # optional

# Now load and cache locally
from datasets import load_dataset
ds = load_dataset("allenai/c4", "fr", split="train", cache_dir=str(CACHE_DIR))
ds=ds