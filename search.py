"""
search.py - find movies by MEANING, optionally narrowed down by exact FILTERS.

Examples:
    uv run python search.py "a feel-good movie for a rainy day"
    uv run python search.py "an old man flies away in his house"
    uv run python search.py "something scary" --min-year 2015 --min-rating 7.5
    uv run python search.py "a story about friendship" --language Telugu
    uv run python search.py "fun film to watch with kids" --max-runtime 100
    uv run python search.py "space adventure" --genre Sci-Fi --k 3
    uv run python search.py "a cosy evening watch" --compare     # MiniLM vs OpenAI side by side

Run `uv run python ingest.py` first, so the database has movies in it.
"""

import argparse

import config
from embedders import get_embedder
from ingest import genre_key, get_collection


# ================================================================ 1. filters -> Chroma's format

def build_filter(min_year=None, max_year=None, min_rating=None,
                 max_runtime=None, genre=None, language=None):
    """
    Turn simple filter values into the dictionary Chroma understands.

    build_filter(min_year=2010)                  -> {"year": {"$gte": 2010}}
    build_filter(min_year=2010, min_rating=8)    -> {"$and": [{"year": {"$gte": 2010}},
                                                              {"rating": {"$gte": 8}}]}
    build_filter()                               -> None   (no filter: search everything)
    """
    conditions = []

    if min_year is not None:
        conditions.append({"year": {"$gte": min_year}})          # $gte = greater than or equal
    if max_year is not None:
        conditions.append({"year": {"$lte": max_year}})          # $lte = less than or equal
    if min_rating is not None:
        conditions.append({"rating": {"$gte": min_rating}})
    if max_runtime is not None:
        conditions.append({"runtime_min": {"$lte": max_runtime}})
    if genre is not None:
        conditions.append({genre_key(genre): True})              # "Sci-Fi" -> {"genre_sci_fi": True}
    if language is not None:
        conditions.append({"language": language.title()})        # "tamil" -> "Tamil"

    if not conditions:
        return None                    # nothing asked for
    if len(conditions) == 1:
        return conditions[0]           # Chroma wants a single condition on its own
    return {"$and": conditions}        # several: ALL must be true


# ================================================================ 2. the search itself

def search(query, model_name=config.DEFAULT_MODEL, k=config.TOP_K, **filters):
    """
    Return the k movies closest in meaning to `query`, best first.
    Each result is a dict: {"title", "year", "rating", "genres", "language",
                            "runtime_min", "similarity", "text"}
    """
    # a) Turn the question into a vector WITH THE SAME MODEL used for the movies
    embedder = get_embedder(model_name)
    query_vector = embedder.embed([query])[0]      # embed() takes a list; we take the one vector back

    # b) Ask Chroma for the nearest movies (that also pass the filters)
    collection = get_collection(model_name)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where=build_filter(**filters),
        include=["metadatas", "documents", "distances"],
    )

    # c) Chroma answers in "lists of lists" (one inner list per query).
    #    We sent ONE query, so we take [0] of each, then walk through them together.
    movies = []
    for movie_id, metadata, text, distance in zip(results["ids"][0], 
                                        results["metadatas"][0],
                                        results["documents"][0],
                                        results["distances"][0]):
        movies.append({
            "id": movie_id,
            "title": metadata["title"],
            "year": metadata["year"],
            "rating": metadata["rating"],
            "runtime_min": metadata["runtime_min"],
            "genres": metadata["genres"],
            "language": metadata["language"],
            # Chroma gives cosine DISTANCE (lower = closer).
            # 1 - distance = cosine SIMILARITY (higher = closer), easier to read.
            "similarity": round(1 - distance, 3),
            "text": text,
        })
    return movies


# ================================================================ 3. printing

def print_results(movies, model_name):
    print(f"\n--- {model_name} ---")
    if not movies:
        print("  No movies matched. Try loosening the filters.")
        return
    for rank, movie in enumerate(movies, start=1):
        print(f"  {rank}. [{movie['similarity']:.2f}]  {movie['title']} ({movie['year']})"
              f"  ★ {movie['rating']}  ·  {movie['runtime_min']} min  ·  {movie['language']}")
        print(f"       {movie['genres']}")


# ================================================================ 4. command line

def main():
    parser = argparse.ArgumentParser(description="Search the movie catalog by meaning.")
    parser.add_argument("query", help='what you are looking for, in quotes')
    parser.add_argument("--model", choices=list(config.MODELS), default=config.DEFAULT_MODEL)
    parser.add_argument("--compare", action="store_true", help="run every available model")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="number of results")
    parser.add_argument("--min-year", type=int)
    parser.add_argument("--max-year", type=int)
    parser.add_argument("--min-rating", type=float)
    parser.add_argument("--max-runtime", type=int, help="in minutes")
    parser.add_argument("--genre", help="e.g. Horror, Sci-Fi, Animation")
    parser.add_argument("--language", help="e.g. English, Tamil, Hindi, Japanese")
    args = parser.parse_args()

    filters = {
        "min_year": args.min_year,
        "max_year": args.max_year,
        "min_rating": args.min_rating,
        "max_runtime": args.max_runtime,
        "genre": args.genre,
        "language": args.language,
    }

    print(f'Query: "{args.query}"')
    active = {name: value for name, value in filters.items() if value is not None}
    if active:
        print(f"Filters: {active}")

    models = config.available_models() if args.compare else [args.model]
    for model_name in models:
        movies = search(args.query, model_name=model_name, k=args.k, **filters)
        print_results(movies, model_name)


if __name__ == "__main__":
    main()