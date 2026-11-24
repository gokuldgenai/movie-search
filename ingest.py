"""
ingest.py - loads the movies, turns them into vectors, and saves them in ChromaDB.

Run:
    uv run python ingest.py                 # every available model
    uv run python ingest.py --model minilm  # just one model
    uv run python ingest.py --reset         # delete and rebuild from scratch

Safe to run again and again: movies are stored by their id (m001, m002...),
so re-running UPDATES them instead of creating duplicates.
"""

import argparse
import json

import chromadb

import config
from embedders import get_embedder


# ================================================================ 1. load the data

def load_movies():
    """Read data/movies.json and return it as a list of dictionaries."""
    with open(config.DATA_FILE, encoding="utf-8") as file:
        return json.load(file)


# ================================================================ 2. one movie -> text + metadata

def movie_to_text(movie):
    """
    Build the sentence we EMBED. It should hold everything about the
    movie's *meaning*: what it's about, how it feels, what kind of film it is.
    """
    return (
        f"{movie['title']}. "
        f"Genres: {', '.join(movie['genres'])}. "
        f"Mood: {movie['mood']}. "
        f"{movie['description']}"
    )


def genre_key(genre):
    """'Sci-Fi' -> 'genre_sci_fi'. Used as a metadata field name."""
    return "genre_" + genre.lower().replace("-", "_").replace(" ", "_")


def movie_to_metadata(movie):
    """
    Build the METADATA: exact facts we want to FILTER on, not search by meaning.
    Chroma metadata values must be simple: text, numbers or True/False.
    """
    metadata = {
        "title": movie["title"],
        "year": movie["year"],
        "runtime_min": movie["runtime_min"],
        "rating": movie["rating"],
        "language": movie["language"],
        "genres": ", ".join(movie["genres"]),   # for display: "Sci-Fi, Drama, Adventure"
    }
    # Chroma can't filter inside a list, so each genre becomes its own True flag:
    #   "genre_sci_fi": True, "genre_drama": True, "genre_adventure": True
    for genre in movie["genres"]:
        metadata[genre_key(genre)] = True
    return metadata


# ================================================================ 3. the database

def get_collection(model_name):
    """
    Open (or create) the Chroma collection for one model.
    Each model gets its own collection, because vectors from different
    models can't be compared.
    """
    client = chromadb.PersistentClient(path=str(config.DB_DIR))   # saved to disk
    return client.get_or_create_collection(
        name=config.MODELS[model_name]["collection"],
        metadata={"hnsw:space": config.DISTANCE},   # use cosine, not Chroma's default L2
        embedding_function=None,                    # we make the vectors ourselves
    )


def delete_collection(model_name):
    """Remove a model's collection completely (used by --reset)."""
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    name = config.MODELS[model_name]["collection"]
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
        print(f"  deleted old collection '{name}'")


# ================================================================ 4. put it all together

def ingest(model_name, movies):
    """Embed every movie with one model and save it in that model's collection."""
    print(f"\n=== {model_name}: {config.MODELS[model_name]['model_name']} ===")

    # a) Prepare four lists, all in the same order (position 0 = the first movie, etc.)
    ids = [movie["id"] for movie in movies]
    texts = [movie_to_text(movie) for movie in movies]
    metadatas = [movie_to_metadata(movie) for movie in movies]

    # b) Turn every text into a vector (this is Step 3's code at work)
    print(f"  embedding {len(texts)} movies...")
    embedder = get_embedder(model_name)
    vectors = embedder.embed(texts)

    # c) Save everything. upsert = add new ids, overwrite existing ones.
    collection = get_collection(model_name)
    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"  saved. Collection '{collection.name}' now holds {collection.count()} movies.")
    return collection


def show_example(collection, movie_id="m001"):
    """Read one movie back from Chroma to prove it was stored."""
    stored = collection.get(ids=[movie_id], include=["documents", "metadatas", "embeddings"])
    print(f"\n  Example stored item ({movie_id}):")
    print(f"    document : {stored['documents'][0][:90]}...")
    print(f"    metadata : {stored['metadatas'][0]}")
    print(f"    embedding: {len(stored['embeddings'][0])} numbers, starting "
          f"{[round(float(x), 3) for x in stored['embeddings'][0][:3]]}")


def main():
    parser = argparse.ArgumentParser(description="Load movies into ChromaDB.")
    parser.add_argument("--model", choices=list(config.MODELS),
                        help="only this model (default: every available model)")
    parser.add_argument("--reset", action="store_true",
                        help="delete the collection first and rebuild it")
    args = parser.parse_args()

    models = [args.model] if args.model else config.available_models()
    movies = load_movies()
    print(f"Loaded {len(movies)} movies from {config.DATA_FILE.name}")

    for model_name in models:
        if args.reset:
            delete_collection(model_name)
        collection = ingest(model_name, movies)
        show_example(collection)


if __name__ == "__main__":
    main()