"""
embedders.py - turns text into vectors (embeddings).

Two embedders with the SAME interface:

    embedder = get_embedder("minilm")        # or "openai"
    vectors = embedder.embed(["text one", "text two"])
    # -> a list with one vector (a list of floats) per text

Because both classes have an embed() method that takes and returns the
same things, the rest of the project never needs to know which model is
being used. Swapping models = changing one name.

Run this file directly to see embeddings in action:
    uv run python embedders.py
"""

import time

import numpy as np

import config


# ====================================================================== local model

class SentenceTransformerEmbedder:
    """Runs a HuggingFace sentence-transformers model on your own computer."""

    def __init__(self, model_name):
        # Imported here instead of at the top because it loads PyTorch,
        # which takes a few seconds. This way the cost is only paid when
        # this embedder is actually used.
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        # First run downloads the model (~90 MB) to a cache folder; after that it's offline.
        self.model = SentenceTransformer(model_name)

    def embed(self, texts, batch_size=32):
        """Embed a list of texts. Returns a list of vectors."""
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,             # how many texts go through the model at once
            normalize_embeddings=True,         # make every vector length 1 (see Concept 2)
            show_progress_bar=len(texts) > 100,
        )
        # encode() returns a NumPy array; Chroma wants plain Python lists.
        return vectors.tolist()


# ====================================================================== OpenAI API

class OpenAIEmbedder:
    """Calls OpenAI's embeddings API. Needs OPENAI_API_KEY in your .env file."""

    def __init__(self, model_name, api_key):
        from openai import OpenAI

        self.name = model_name
        self.client = OpenAI(api_key=api_key)

    def embed(self, texts, batch_size=100, max_retries=3):
        """
        Embed a list of texts by sending them to the API in batches.

        Sending 100 texts in one request is far faster and cheaper on
        overhead than 100 separate requests. If the API is busy (rate
        limit) or the network hiccups, we wait and try again.
        """
        from openai import APIConnectionError, RateLimitError

        all_vectors = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            for attempt in range(1, max_retries + 1):
                try:
                    response = self.client.embeddings.create(model=self.name, input=batch)
                    break                                    # success: leave the retry loop
                except (RateLimitError, APIConnectionError) as error:
                    if attempt == max_retries:
                        raise                                # out of retries: give up loudly
                    wait = 2 ** attempt                      # 2s, then 4s ("exponential backoff")
                    print(f"  OpenAI error ({type(error).__name__}), retrying in {wait}s...")
                    time.sleep(wait)

            # Each item in response.data has an .index saying which input it belongs to.
            # Sorting by it guarantees vector order matches text order.
            for item in sorted(response.data, key=lambda d: d.index):
                all_vectors.append(item.embedding)

        # OpenAI embeddings already come back with length 1, so no normalizing needed.
        return all_vectors


# ====================================================================== factory

_loaded = {}   # embedders we've already created, so each model loads only once


def get_embedder(name=config.DEFAULT_MODEL):
    """
    Return the embedder for a model name from config.MODELS ("minilm", "openai").

    This is called a FACTORY function: callers ask for an embedder by name,
    and this function decides which class to build and how.
    """
    if name not in config.MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(config.MODELS)}")

    if name not in _loaded:
        settings = config.MODELS[name]

        if settings["provider"] == "sentence-transformers":
            _loaded[name] = SentenceTransformerEmbedder(settings["model_name"])

        elif settings["provider"] == "openai":
            if not config.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
            _loaded[name] = OpenAIEmbedder(settings["model_name"], config.OPENAI_API_KEY)

        else:
            raise ValueError(f"Unknown provider '{settings['provider']}'")

    return _loaded[name]


# ====================================================================== demo

if __name__ == "__main__":
    sentences = [
        "A boy discovers he is a wizard and goes to a school of magic.",            # 0
        "An orphan learns he has magical powers and attends a witchcraft academy.",  # 1  same meaning as 0
        "A family must stay completely silent to survive monsters that hunt by sound.",  # 2  different
    ]

    for name in config.available_models():
        settings = config.MODELS[name]
        print(f"\n=== {name}: {settings['model_name']} ===")

        embedder = get_embedder(name)
        vectors = np.array(embedder.embed(sentences))      # shape: (3 texts, N numbers)

        print(f"Shape: {vectors.shape}  (expected {len(sentences)} x {settings['dimensions']})")
        print(f"First 5 numbers of sentence 0: {np.round(vectors[0][:5], 3)}")
        print(f"Length of each vector: {np.round(np.linalg.norm(vectors, axis=1), 3)}  (1.0 = normalized)")

        # Vectors are length 1, so dot product = cosine similarity.
        # vectors @ vectors.T compares every sentence with every other one at once.
        similarity = vectors @ vectors.T
        print("Cosine similarity matrix (rows/cols = sentences 0, 1, 2):")
        print(np.round(similarity, 2))
        print(f"0 vs 1 (same meaning): {similarity[0, 1]:.2f}   "
              f"0 vs 2 (different): {similarity[0, 2]:.2f}")