#!/usr/bin/env python3
"""Wikipedia evidence retrieval and TF-IDF sentence ranking with local caching.
Author: Aryan Goel
"""

__author__ = "Aryan Goel"

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

session = requests.Session()
session.headers.update({
    "User-Agent": "FEVERBenchmarkStudentProject/1.0"
})

WIKI_API = "https://en.wikipedia.org/w/api.php"


def load_cache(cache_path: Path) -> Dict[str, str]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load cache: %s. Starting fresh.", exc)
    return {}


def save_cache(cache: Dict[str, str], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


import time


def wikipedia_search(claim: str, limit: int = 3, timeout: int = 30) -> List[str]:
    """Find Wikipedia article titles related to the claim with rate-limiting and retry."""
    for attempt in range(3):
        try:
            time.sleep(0.3)
            response = session.get(
                WIKI_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": claim,
                    "srlimit": limit,
                    "format": "json",
                },
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return [item["title"] for item in data.get("query", {}).get("search", [])]
        except requests.RequestException as exc:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
            else:
                logger.warning("Wikipedia search failed for claim '%s': %s", claim[:50], exc)
    return []


def get_wikipedia_text(title: str, cache: Dict[str, str], timeout: int = 30) -> str:
    """Fetch plain text of Wikipedia article using MediaWiki API extracts with retry."""
    if title in cache:
        return cache[title]

    for attempt in range(3):
        try:
            time.sleep(0.3)
            response = session.get(
                WIKI_API,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exsectionformat": "plain",
                    "titles": title,
                    "format": "json",
                    "redirects": 1,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})
            page = next(iter(pages.values()))
            text = page.get("extract", "")
            cache[title] = text
            return text
        except requests.RequestException as exc:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
            else:
                logger.warning("Wikipedia text fetch failed for '%s': %s", title, exc)
    return ""


def split_sentences(text: str) -> List[str]:
    """Splits article text into clean sentences."""
    cleaned = re.sub(r"\s+", " ", text)
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", cleaned)
        if len(s.strip()) > 30
    ]


def rank_sentences_tfidf(claim: str, candidates: List[Dict[str, str]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Rank candidate sentences by TF-IDF cosine similarity against the claim."""
    if not candidates:
        return []

    texts = [claim] + [c["sentence"] for c in candidates]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
    )

    try:
        matrix = vectorizer.fit_transform(texts)
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    except ValueError:
        return []

    ranked_indices = similarities.argsort()[::-1]
    selected = []

    for idx in ranked_indices:
        sim = similarities[idx]
        if sim <= 0:
            continue
        item = candidates[idx]
        selected.append({
            "title": item["title"],
            "sentence": item["sentence"],
            "similarity": round(float(sim), 4),
        })
        if len(selected) >= top_k:
            break

    return selected


def retrieve_evidence(
    claim: str,
    cache: Dict[str, str],
    top_k: int = 5,
    search_limit: int = 3,
) -> List[Dict[str, Any]]:
    """Retrieve Wikipedia articles and return top-k most similar sentences."""
    titles = wikipedia_search(claim, limit=search_limit)
    candidates = []

    for title in titles:
        text = get_wikipedia_text(title, cache)
        if text:
            for sentence in split_sentences(text):
                candidates.append({"title": title, "sentence": sentence})

    return rank_sentences_tfidf(claim, candidates, top_k=top_k)


if __name__ == "__main__":
    test_claim = "The Roman Empire fell in 476 AD."
    test_cache = {}
    ev = retrieve_evidence(test_claim, test_cache, top_k=3)
    print(f"Evidence for: '{test_claim}'")
    print(json.dumps(ev, indent=2))
