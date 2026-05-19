"""Throwaway PubMed demo fetcher; replaced by the async harvester in V1-S03."""

from __future__ import annotations

from typing import Any

from Bio import Entrez


def _safe_str(value: Any) -> str:
    """Cast a Biopython Entrez node to str, tolerating None."""
    if value is None:
        return ""
    return str(value)


def _extract_year(pub_date: Any) -> int | None:
    """Pull a 4-digit year from a PubDate node, returning None when unavailable."""
    try:
        year_raw = pub_date.get("Year") if hasattr(pub_date, "get") else None
    except Exception:
        return None
    if year_raw is None:
        return None
    try:
        return int(str(year_raw))
    except (TypeError, ValueError):
        return None


def _extract_record(record: Any) -> dict[str, object]:
    """Pull the demo fields out of a single PubmedArticle record."""
    medline = record.get("MedlineCitation", {}) if hasattr(record, "get") else {}
    article = medline.get("Article", {}) if hasattr(medline, "get") else {}

    pmid = _safe_str(medline.get("PMID", "")) if hasattr(medline, "get") else ""
    title = _safe_str(article.get("ArticleTitle", "")) if hasattr(article, "get") else ""

    abstract_text = ""
    has_abstract = False
    try:
        abstract_node = article.get("Abstract", {}) if hasattr(article, "get") else {}
        fragments = abstract_node.get("AbstractText", []) if hasattr(abstract_node, "get") else []
        if fragments:
            abstract_text = " ".join(_safe_str(fragment) for fragment in fragments).strip()
            has_abstract = bool(abstract_text)
    except Exception:
        abstract_text = ""
        has_abstract = False

    journal_title = ""
    year: int | None = None
    try:
        journal_node = article.get("Journal", {}) if hasattr(article, "get") else {}
        journal_title = (
            _safe_str(journal_node.get("Title", "")) if hasattr(journal_node, "get") else ""
        )
        issue = journal_node.get("JournalIssue", {}) if hasattr(journal_node, "get") else {}
        pub_date = issue.get("PubDate", {}) if hasattr(issue, "get") else {}
        year = _extract_year(pub_date)
    except Exception:
        journal_title = journal_title or ""
        year = None

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract_text,
        "journal": journal_title,
        "year": year,
        "has_abstract": has_abstract,
    }


def fetch_demo_papers(
    journal: str,
    year_range: tuple[int, int],
    max_papers: int,
    email: str,
) -> list[dict[str, object]]:
    """Fetch a small batch of PubMed records for the demo pipeline."""
    _entrez: Any = Entrez
    _entrez.email = email
    y0, y1 = year_range
    term = f'"{journal}"[Journal] AND {y0}:{y1}[PDAT]'

    search_handle = Entrez.esearch(db="pubmed", term=term, retmax=max_papers)
    try:
        search_result: Any = Entrez.read(search_handle)
    finally:
        search_handle.close()

    pmids: list[str] = [str(pid) for pid in search_result.get("IdList", [])]
    if not pmids:
        return []

    fetch_handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="xml",
        retmode="xml",
    )
    try:
        records: Any = Entrez.read(fetch_handle)
    finally:
        fetch_handle.close()

    articles = records.get("PubmedArticle", []) if hasattr(records, "get") else []
    rows: list[dict[str, object]] = []
    for record in articles:
        try:
            rows.append(_extract_record(record))
        except Exception:
            continue
    return rows
