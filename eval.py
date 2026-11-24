"""
evaluate.py - measures how good each model's search is, with numbers.

It runs 30 test questions (data/eval_queries.json). For each one we already
know which movie(s) are the right answer. We check where those movies
appear in the results, and turn that into two scores:

    Recall@5  - "did the right movies show up in the top 5?"   (0 to 1, higher = better)
    MRR       - "how high was the FIRST right movie?"           (0 to 1, higher = better)

Run:
    uv run python evaluate.py              # every available model
    uv run python evaluate.py --details    # also show every question's result
"""

import argparse
import json
import time

import config
from search import search

K = 5   # we judge the top 5 results


# ================================================================ 1. the two scores for ONE question

def recall_at_k(found_ids, relevant_ids):
    """
    What share of the right movies appear in the results?
      right = [Up]              found = [Coco, Up, ...]  -> 1 of 1 -> 1.0
      right = [Get Out, It, ...3 movies]  found 2 of them -> 2/3 -> 0.67
    """
    hits = [movie_id for movie_id in relevant_ids if movie_id in found_ids]
    return len(hits) / len(relevant_ids)


def reciprocal_rank(found_ids, relevant_ids):
    """
    1 / (position of the FIRST right movie). 0 if none was found.
      first right movie at rank 1 -> 1/1 = 1.0
      first right movie at rank 2 -> 1/2 = 0.5
      first right movie at rank 4 -> 1/4 = 0.25
      not in the top 5            -> 0
    """
    for rank, movie_id in enumerate(found_ids, start=1):
        if movie_id in relevant_ids:
            return 1 / rank
    return 0.0


# ================================================================ 2. run every question for one model

def evaluate_model(model_name, questions):
    """Run all questions with one model; return one result row per question."""
    rows = []
    start = time.perf_counter()                       # start a stopwatch

    for question in questions:
        results = search(question["query"], model_name=model_name, k=K)
        found_ids = [movie["id"] for movie in results]

        rows.append({
            "query": question["query"],
            "type": question["type"],
            "recall": recall_at_k(found_ids, question["relevant"]),
            "rr": reciprocal_rank(found_ids, question["relevant"]),
            "top_title": results[0]["title"] if results else "-",
        })

    seconds = time.perf_counter() - start            # stop the stopwatch
    return rows, seconds


def average(numbers):
    return sum(numbers) / len(numbers) if numbers else 0.0


# ================================================================ 3. printing

def print_details(model_name, rows):
    print(f"\n--- {model_name}: every question ---")
    print(f"  {'recall':>6} {'RR':>5}  {'question':<62} top result")
    for row in rows:
        flag = "  " if row["rr"] == 1 else "⚠ "          # mark questions not answered perfectly
        print(f"{flag}{row['recall']:>6.2f} {row['rr']:>5.2f}  {row['query'][:60]:<62} {row['top_title']}")


def print_summary(summary):
    print("\n=== Summary (higher is better) ===")
    print(f"{'model':<8} {'Recall@5':>9} {'MRR':>6} {'plot MRR':>9} {'mood MRR':>9} {'ms/query':>9}")
    for model_name, s in summary.items():
        print(f"{model_name:<8} {s['recall']:>9.2f} {s['mrr']:>6.2f} "
              f"{s['plot_mrr']:>9.2f} {s['mood_mrr']:>9.2f} {s['ms']:>9.0f}")


# ================================================================ 4. main

def main():
    parser = argparse.ArgumentParser(description="Score each model on the test questions.")
    parser.add_argument("--details", action="store_true", help="show every question's result")
    args = parser.parse_args()

    with open(config.EVAL_FILE, encoding="utf-8") as file:
        questions = json.load(file)
    print(f"Evaluating on {len(questions)} test questions, top {K} results each")

    summary = {}
    for model_name in config.available_models():
        rows, seconds = evaluate_model(model_name, questions)
        if args.details:
            print_details(model_name, rows)

        summary[model_name] = {
            "recall": average([r["recall"] for r in rows]),
            "mrr": average([r["rr"] for r in rows]),
            "plot_mrr": average([r["rr"] for r in rows if r["type"] == "plot"]),
            "mood_mrr": average([r["rr"] for r in rows if r["type"] == "mood"]),
            "ms": seconds / len(questions) * 1000,        # milliseconds per question
        }

    print_summary(summary)


if __name__ == "__main__":
    main()