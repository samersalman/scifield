"""Topic-coherence helpers (NPMI + C_v) for the thematic backbone.

V1-S06's Gate G1 threshold is NPMI ≥ 0.18, computed against the deduped
abstract corpus tokenised to lower-case alnum tokens, top-10 words per
topic, with the BERTopic noise topic excluded. This module exposes the
two primitives that wrap Gensim's :class:`gensim.models.CoherenceModel`
to make that calculation deterministic and dependency-light at import
time.

We import ``gensim`` lazily inside :func:`compute_coherence` because
Gensim historically pins ``numpy<2`` and we want this module to remain
importable for tests that only need :func:`tokenise_for_coherence`.
"""

from __future__ import annotations

import re

__all__ = [
    "compute_coherence",
    "tokenise_for_coherence",
]


# A deliberately small English stop-word list: large list ≠ better here.
# We strip common function words that drown the c-TF-IDF top-N; medical
# vocabulary (e.g. "patient", "study") stays in because excluding it
# would bias NPMI against domain-realistic topics.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "we",
        "our",
        "their",
        "they",
        "these",
        "those",
        "than",
        "then",
        "there",
        "which",
        "who",
        "whom",
        "whose",
        "if",
        "so",
        "such",
        "not",
        "no",
        "yes",
        "do",
        "does",
        "did",
        "been",
        "being",
        "between",
        "among",
        "also",
        "however",
        "thus",
        "may",
        "can",
        "could",
        "should",
        "would",
        "might",
        "must",
        "shall",
    }
)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenise_for_coherence(documents: list[str]) -> list[list[str]]:
    """Lower-case alnum tokenisation with a small English stop-word filter.

    Deterministic and stdlib-only — no NLTK download, no spaCy model. The
    goal is to match the tokeniser BERTopic's :class:`CountVectorizer`
    uses for the topic word lists, so coherence scores are computed on
    the same vocabulary space the topics were learned on.
    """
    out: list[list[str]] = []
    for doc in documents:
        if not doc:
            out.append([])
            continue
        tokens = _TOKEN_RE.findall(doc.lower())
        out.append([t for t in tokens if len(t) >= 2 and t not in _STOPWORDS])
    return out


def _clean_word_lists(topic_word_lists: list[list[str]], top_n: int) -> list[list[str]]:
    """Drop empties and truncate to ``top_n`` — belt-and-braces guard.

    Callers should already strip the BERTopic noise topic (-1) before
    calling, but Gensim raises confusing errors on an empty inner list,
    so we defensively skip those here too.
    """
    cleaned: list[list[str]] = []
    for words in topic_word_lists:
        trimmed = [w for w in words[:top_n] if w]
        if trimmed:
            cleaned.append(trimmed)
    return cleaned


def compute_coherence(
    topic_word_lists: list[list[str]],
    texts: list[list[str]],
    *,
    top_n: int = 10,
    metrics: tuple[str, ...] = ("c_npmi", "c_v"),
) -> dict[str, float]:
    """Return the mean coherence across topics for each requested metric.

    Parameters
    ----------
    topic_word_lists:
        Top words per topic in BERTopic order. The noise topic (-1) is
        expected to be stripped by the caller, but an empty inner list is
        tolerated and skipped.
    texts:
        Tokenised reference corpus aligned with the texts the topics were
        fit on. Usually the output of :func:`tokenise_for_coherence`.
    top_n:
        Truncate each topic to this many words before computing.
    metrics:
        Gensim coherence identifiers; one CoherenceModel is built per
        metric since ``coherence=`` differs.
    """
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    cleaned = _clean_word_lists(topic_word_lists, top_n)
    if not cleaned:
        return {m: float("nan") for m in metrics}

    dictionary = Dictionary(texts)
    out: dict[str, float] = {}
    for metric in metrics:
        cm = CoherenceModel(
            topics=cleaned,
            texts=texts,
            dictionary=dictionary,
            coherence=metric,
            topn=top_n,
        )
        out[metric] = float(cm.get_coherence())
    return out
