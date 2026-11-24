# 🎬 Movie Search: Semantic Search over Structured Data

Search a movie catalog by **what you feel like watching**, not by exact titles or keywords.

> *"an elderly widower travels to South America in his floating house"* → **Up**
> *"astronauts fighting to survive in space"* → **Interstellar, The Martian, Gravity**

The project embeds structured movie records with two embedding models (a free local sentence-transformers model and OpenAI's `text-embedding-3-small`), stores them in **ChromaDB**, and combines **semantic search** with **exact metadata filters** (year, rating, runtime, genre, language). A labelled evaluation set measures which model retrieves better, using **Recall@5** and **MRR**.

Built from scratch in plain Python: **no LangChain**, so every step (embedding, storing, querying, filtering, scoring) is visible in the code.

---

## Contents

- [Features](#features)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Design decisions](#design-decisions)
- [Evaluation results](#evaluation-results)
- [Limitations](#limitations)
- [Next steps](#next-steps)

---

## Features

- **Semantic search:** finds movies by meaning, even when the query shares no words with the description.
- **Hybrid filtering:** meaning comes from embeddings; hard rules (year ≥ 2015, rating ≥ 8, runtime ≤ 100 min, genre, language) come from Chroma metadata filters.
- **Two embedding models, side by side:** local `all-MiniLM-L6-v2` (free, private, offline) vs OpenAI `text-embedding-3-small` (hosted API), each in its own collection.
- **Measured, not guessed:** a 30-question labelled test set scores both models with Recall@5 and MRR, split by query type.
- **Idempotent ingestion:** stable ids plus `upsert` mean re-running never creates duplicates.
- **Two interfaces, one search function:** a command-line tool and a Streamlit web app that both call the same `search()`.

---

## Tech stack

| Purpose | Tool |
| --- | --- |
| Package management | [uv](https://docs.astral.sh/uv/) |
| Vector database | [ChromaDB](https://www.trychroma.com/) (persistent, cosine distance, HNSW index) |
| Local embeddings | [sentence-transformers](https://www.sbert.net/): `all-MiniLM-L6-v2` (384 dimensions) |
| Hosted embeddings | OpenAI `text-embedding-3-small` (1,536 dimensions) |
| Web UI | [Streamlit](https://streamlit.io/) |
| Secrets | python-dotenv |

---

## Project structure

```
movie-search/
├── data/
│   ├── movies.json          # 64 movies: title, year, runtime, rating, genres, language, mood, description
│   └── eval_queries.json    # 30 test questions with their correct movie ids
├── config.py                # all settings: paths, model registry, top-k, distance metric
├── embedders.py             # text -> vectors, for both models, behind one embed() interface
├── ingest.py                # movie -> text + metadata -> vectors -> ChromaDB
├── search.py                # semantic search + metadata filters (command line)
├── evaluate.py              # Recall@5 and MRR for each model
├── app.py                   # Streamlit web app
├── pyproject.toml / uv.lock # dependencies (managed by uv)
└── .env                     # your OPENAI_API_KEY (not committed)
```

`chroma_db/` is created on first ingest and is git-ignored. It can always be rebuilt.

---

## Getting started

**Requirements:** Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <your-repo-url>
cd movie-search
uv sync                                  # installs exactly the versions in uv.lock
```

**Optional:** to use the OpenAI model, create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Without a key, everything still works with the local MiniLM model only.

> The first run downloads `all-MiniLM-L6-v2` (about 90 MB) from Hugging Face. After that it runs offline.

---

## Usage

### 1. Load the movies into ChromaDB

```bash
uv run python ingest.py                  # every available model
uv run python ingest.py --model minilm   # one model only
uv run python ingest.py --reset          # delete and rebuild (after changing the model or the text format)
```

### 2. Search from the command line

```bash
uv run python search.py "a feel-good movie for a rainy day"
uv run python search.py "something scary" --min-year 2015 --min-rating 7.5
uv run python search.py "a story about friendship" --language Telugu
uv run python search.py "fun film to watch with kids" --max-runtime 100 --genre Animation
uv run python search.py "inspiring true story" --compare        # both models
```

| Option | Meaning |
| --- | --- |
| `--k N` | Number of results (default 5) |
| `--min-year`, `--max-year` | Year range |
| `--min-rating` | Minimum rating |
| `--max-runtime` | Maximum length in minutes |
| `--genre` | e.g. `Horror`, `Sci-Fi`, `Animation` |
| `--language` | e.g. `English`, `Tamil`, `Hindi`, `Telugu`, `Japanese`, `Korean`, `Spanish` |
| `--model` | `minilm` (default) or `openai` |
| `--compare` | Run every available model |

### 3. Evaluate the models

```bash
uv run python evaluate.py                # summary table
uv run python evaluate.py --details      # every question, with ⚠ on imperfect answers
```

### 4. Launch the web app

```bash
uv run streamlit run app.py
```

Then open http://localhost:8501. Filters are in the sidebar; "compare both" shows the two models side by side.

---

## How it works

```
INGEST (once, or whenever the data changes)

  movies.json
      │
      ├─► movie_to_text()      "Up. Genres: Animation, Adventure. Mood: heartwarming...
      │                          A grumpy widower ties thousands of balloons to his house..."
      │                          └─► embed() ─► vector (384 or 1,536 numbers)
      │
      └─► movie_to_metadata()  {year: 2009, rating: 8.3, runtime_min: 96, language: "English",
                                 genre_animation: True, genre_adventure: True, ...}
      │
      ▼
  ChromaDB collection (one per model): id + vector + text + metadata, saved to disk


SEARCH (every question)

  "an old man flies away in his house"
      │
      ├─► embed() with the SAME model ─► query vector
      ├─► build_filter()  ─► {"$and": [{"year": {"$gte": 2000}}, ...]}   (optional)
      ▼
  collection.query()  ─► keep movies passing the filter ─► rank by cosine similarity ─► top k
```

---

## Design decisions

**1. Split each record into "meaning" and "facts".**
Title, genres, mood and description are combined into one text and **embedded**. Year, rating, runtime and language are stored as **metadata** and **filtered** exactly. Embeddings are good at meaning but unreliable with numbers: a query like "a horror movie made after 2015" can still return older films. A metadata filter enforces it exactly.

**2. One-hot genre flags.**
Chroma metadata values must be scalars (text, numbers, booleans), so a genre list can't be filtered directly. Each genre becomes its own boolean field (`genre_horror: True`), which turns "movies that include Horror" into a simple equality filter.

**3. One collection per embedding model.**
Vectors from different models live in different spaces and are not comparable. Each model gets its own collection (`movies_minilm`, `movies_openai`), and queries are always embedded with the same model as the collection.

**4. Embeddings computed explicitly.**
Collections are created with `embedding_function=None`, and the code passes vectors to Chroma itself. This keeps the embedding step visible, makes it easy to swap models, and allows batching and retries for the API model.

**5. Cosine distance, set on purpose.**
Chroma defaults to L2 distance. The collections use `hnsw:space = cosine`, and vectors are normalized to length 1. Chroma returns cosine *distance*, which is converted to similarity (1 − distance) for display.

**6. Stable ids plus `upsert`.**
Each movie keeps its id (`m001` to `m064`), so re-running ingest updates records instead of duplicating them.

**7. One search function, many interfaces.**
`search()` holds all the retrieval logic. The CLI, the evaluation script and the Streamlit app all call it, so what gets evaluated is exactly what users get.

**8. Production habits in the API embedder.**
Batching (100 texts per request), retries with exponential backoff on rate-limit and connection errors, and re-ordering results by index so vectors always match their texts.

---

## Evaluation results

30 hand-labelled questions, top 5 results each. Questions deliberately avoid the descriptions' own wording, so they test meaning rather than keyword overlap.

- **plot** (20 questions): describe one specific movie, with 1 correct answer. *"a worried father fish searches the sea for his missing son"*
- **mood** (10 questions): describe a theme or feeling, with 2 or 3 correct answers. *"true stories filmed in real life, not fiction"*

| Model | Recall@5 | MRR | Plot MRR | Mood MRR |
| --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 (local) | 0.84 | 0.81 | 0.90 | 0.63 |
| text-embedding-3-small (OpenAI) | **0.94** | **0.98** | **0.97** | **1.00** |

**Recall@5** is the share of correct movies that appear in the top 5. **MRR** (Mean Reciprocal Rank) is the average of 1 ÷ the rank of the first correct movie. Both range from 0 to 1, and higher is better.

**Findings**

- Both models are strong on **plot** lookups (0.90 vs 0.97 MRR).
- The difference is almost entirely on **mood** queries (0.63 vs 1.00). The smaller local model struggles more with abstract themes, such as linking "real life, not fiction" to documentaries.
- **Trade-off:** MiniLM is free, private, runs offline and avoids a network round-trip per query. OpenAI costs money per token and sends data to an external API, but ranks noticeably better for recommendation-style queries.
- **Recommendation:** for "recommend me something like…" features, the hosted model's quality gain is worth it. For exact lookups, or where data can't leave the machine, MiniLM is good enough.

Latency is printed by `evaluate.py` (ms/query, measured after a warm-up search) and depends on your hardware and network.

---

## Limitations

- **Small test set.** With 10 mood questions, a single question changes mood MRR by 0.10. Large gaps (0.63 vs 1.00) are meaningful; small ones are not.
- **Same author for data and questions.** A stronger evaluation would use real user queries written independently of the catalog.
- **Small catalog.** 64 movies is enough to learn and compare, but too small for meaningful speed benchmarks.
- **Known embedding weaknesses:** negation ("a movie that is *not* scary") and numbers in free text ("made after 2015") are handled poorly by embeddings alone. Metadata filters cover the numeric case.
- **Data:** movie descriptions are original short summaries written for this project; ratings are approximate.

---

## Next steps

- **Hybrid search:** combine BM25 keyword scoring with vector scores (Reciprocal Rank Fusion) for exact names and rare terms.
- **FAISS:** benchmark Flat vs HNSW indexes against ChromaDB on a larger corpus.
- **More models:** add `BAAI/bge-small-en-v1.5` or `text-embedding-3-large` by adding one entry to `config.MODELS`.
- **Bigger, independent evaluation set,** with nDCG for graded relevance.
- **RAG:** pass the retrieved movies to an LLM to write a personalised recommendation with reasons.

---

## What this project demonstrates

Vector embeddings · sentence-transformers vs OpenAI embeddings · ChromaDB collections, metadata and CRUD · metadata filtering (`$and`, `$gte`, `$lte`) · cosine similarity and normalization · structured data → text + metadata · retrieval evaluation (Recall@k, MRR) · benchmarking with warm-up · Streamlit · clean separation of logic and interface.