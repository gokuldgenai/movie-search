"""
config.py - every setting for the project lives here.

Other files do `import config` and read values like config.TOP_K,
so changing a setting here changes it everywhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------ paths
# Path(__file__) is the location of THIS file; .parent is the folder it is in.
# Building paths from it means the scripts work no matter which folder you
# run them from.
PROJECT_DIR = Path(__file__).parent
DATA_FILE = PROJECT_DIR / "data" / "movies.json"          # the catalog (Step 1)
EVAL_FILE = PROJECT_DIR / "data" / "eval_queries.json"    # test questions (Step 6)
DB_DIR = PROJECT_DIR / "chroma_db"                        # where Chroma saves data

# ------------------------------------------------------------------ embedding models
# Each model gets its OWN Chroma collection, because vectors from different
# models live on different "maps" and must never be mixed.
MODELS = {
    "minilm": {
        "provider": "sentence-transformers",   # runs locally, free, no API key
        "model_name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "collection": "movies_minilm",
    },
    "openai": {
        "provider": "openai",                  # hosted API, needs OPENAI_API_KEY
        "model_name": "text-embedding-3-small",
        "dimensions": 1536,
        "collection": "movies_openai",
    },
}

DEFAULT_MODEL = "minilm"


# None if you haven't set a key - the OpenAI model is then simply skipped.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------------------------ search settings
TOP_K = 5              # how many movies a search returns
DISTANCE = "cosine"    # Chroma's default is "l2"; we choose cosine on purpose

def available_models():
    """Names of the models we can actually use right now."""
    return [
        name for name, settings in MODELS.items()
        if settings["provider"] != "openai" or OPENAI_API_KEY
    ]