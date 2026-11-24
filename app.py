"""
app.py - a web page for searching the movie catalog.

Run:   uv run streamlit run app.py
Then open the link it prints (usually http://localhost:8501).

HOW STREAMLIT WORKS (the one thing to understand):
Streamlit runs this whole file from top to bottom EVERY time you type,
click or move a slider. Each widget (slider, box, button) simply returns
its current value. So the file reads like a normal script:
"read the filters -> run the search -> draw the results".
"""

import streamlit as st

import config
from embedders import get_embedder
from ingest import load_movies
from search import search

st.set_page_config(page_title="Movie Search", page_icon="🎬", layout="wide")


# ================================================================ 1. load things ONCE

@st.cache_data
def load_catalog():
    """
    Read movies.json once. @st.cache_data remembers the result, so the file
    isn't re-read on every click.
    Returns: movies by id, all genres, all languages, the year range.
    """
    movies = load_movies()
    by_id = {movie["id"]: movie for movie in movies}
    genres = sorted({genre for movie in movies for genre in movie["genres"]})
    languages = sorted({movie["language"] for movie in movies})
    years = (min(m["year"] for m in movies), max(m["year"] for m in movies))
    return by_id, genres, languages, years


@st.cache_resource
def warm_up(model_name):
    """Load the embedding model once (it takes a few seconds the first time)."""
    get_embedder(model_name)
    return True


movies_by_id, all_genres, all_languages, (first_year, last_year) = load_catalog()
models = config.available_models()


# ================================================================ 2. sidebar: settings and filters

st.sidebar.header("Settings")

mode_options = models + (["compare both"] if len(models) > 1 else [])
mode = st.sidebar.radio("Embedding model", mode_options)
k = st.sidebar.slider("Number of results", min_value=1, max_value=10, value=5)

st.sidebar.header("Filters")
st.sidebar.caption("Meaning comes from the embedding. These are exact rules.")

year_range = st.sidebar.slider("Year", first_year, last_year, (first_year, last_year))
min_rating = st.sidebar.slider("Minimum rating", 6.0, 9.5, 6.0, step=0.1)
max_runtime = st.sidebar.slider("Maximum runtime (minutes)", 80, 190, 190, step=5)
genre = st.sidebar.selectbox("Genre", ["Any"] + all_genres)
language = st.sidebar.selectbox("Language", ["Any"] + all_languages)

# Turn the widget values into the filters search() expects.
# "None" means "don't filter on this", so untouched sliders add no rule.
filters = {
    "min_year": year_range[0] if year_range[0] > first_year else None,
    "max_year": year_range[1] if year_range[1] < last_year else None,
    "min_rating": min_rating if min_rating > 6.0 else None,
    "max_runtime": max_runtime if max_runtime < 190 else None,
    "genre": None if genre == "Any" else genre,
    "language": None if language == "Any" else language,
}


# ================================================================ 3. main area: the search box

st.title("🎬 Movie Search")
st.write("Describe what you feel like watching. It searches by **meaning**, not exact words.")

examples = ["a feel-good movie for a rainy day", "an old man flies away in his house",
            "something scary but clever", "inspiring true story"]
st.caption("Try: " + " · ".join(f"*{e}*" for e in examples))

query = st.text_input("What do you want to watch?", placeholder="e.g. a cosy film to watch with family")


# ================================================================ 4. drawing one result

def show_movie(rank, result):
    """Draw one result as a bordered card."""
    movie = movies_by_id[result["id"]]          # full details (mood, description) from movies.json
    with st.container(border=True):
        left, right = st.columns([5, 1])
        left.markdown(f"**{rank}. {movie['title']}** ({movie['year']})")
        left.caption(f"★ {movie['rating']} · {movie['runtime_min']} min · "
                     f"{movie['language']} · {', '.join(movie['genres'])}")
        right.metric("Match", f"{result['similarity']:.2f}")
        st.write(movie["description"])
        st.caption(f"Mood: {movie['mood']}")


def show_results(model_name):
    """Run the search for one model and draw its results."""
    with st.spinner(f"Searching with {model_name}..."):
        warm_up(model_name)
        results = search(query, model_name=model_name, k=k, **filters)
    if not results:
        st.warning("No movies match these filters. Try loosening them in the sidebar.")
    for rank, result in enumerate(results, start=1):
        show_movie(rank, result)


# ================================================================ 5. run the search

if not query:
    st.info("Type something above to start.")
elif mode == "compare both":
    columns = st.columns(len(models))           # one column per model, side by side
    for column, model_name in zip(columns, models):
        with column:
            st.subheader(model_name)
            show_results(model_name)
else:
    show_results(mode)