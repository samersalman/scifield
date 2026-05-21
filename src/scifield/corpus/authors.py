"""Author disambiguation — three-layer (OpenAlex ID -> ORCID -> heuristic).

V1-S04 §5: stitch a stable ``author_canonical_id`` onto every staging row
emitted by the OpenAlex parser. ORCID overrides OpenAlex (more authoritative);
authors lacking both fall through to a deterministic heuristic hash over
``(normalized_last_name, first_initial, dominant_institution_signal)``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

_NON_ALPHA_RE = re.compile(r"[^a-z]+")
_WS_RE = re.compile(r"\s+")


@dataclass(slots=True)
class AuthorsConfig:
    heuristic_salt: str = "scifield_v1"


def normalize_last_name(name: str) -> str:
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    collapsed = _WS_RE.sub(" ", ascii_only).strip()
    return _NON_ALPHA_RE.sub("", collapsed)


def first_initial(display_name: str) -> str:
    if not display_name:
        return ""
    decomposed = unicodedata.normalize("NFKD", display_name)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    for token in ascii_only.split():
        for ch in token:
            if ch.isalpha():
                return ch.lower()
    return ""


def _last_name_token(display_name: str) -> str:
    if not display_name:
        return ""
    tokens = [t for t in display_name.split() if t]
    if not tokens:
        return ""
    return normalize_last_name(tokens[-1])


def most_common_institution_signal(institutions: list[dict]) -> str:
    if not institutions:
        return ""
    oa_ids: list[str] = []
    countries: list[str] = []
    for inst in institutions:
        if not isinstance(inst, dict):
            continue
        oa_id = inst.get("oa_id") or ""
        if oa_id:
            oa_ids.append(oa_id)
        cc = inst.get("country_code") or ""
        if cc:
            countries.append(cc)
    if oa_ids:
        return Counter(oa_ids).most_common(1)[0][0]
    if countries:
        return Counter(countries).most_common(1)[0][0]
    return ""


def heuristic_id(salt: str, last_name: str, initial: str, signal: str) -> str:
    payload = f"{salt}|{last_name}|{initial}|{signal}".encode()
    digest = hashlib.sha1(payload).hexdigest()[:16]
    return f"H{digest}"


def _is_first_label(label: str, position: int, min_pos: int) -> bool:
    if label:
        return label.lower() == "first"
    return position == min_pos


def _is_last_label(label: str, position: int, max_pos: int) -> bool:
    if label:
        return label.lower() == "last"
    return position == max_pos


def disambiguate_authorships(
    staging_rows: list[dict[str, Any]],
    cfg: AuthorsConfig,
) -> list[dict[str, Any]]:
    if not staging_rows:
        return []

    pmid_positions: dict[str, list[int]] = defaultdict(list)
    for row in staging_rows:
        pmid = row.get("pmid") or ""
        pos = int(row.get("author_position") or 0)
        pmid_positions[pmid].append(pos)
    pmid_min: dict[str, int] = {p: min(v) for p, v in pmid_positions.items()}
    pmid_max: dict[str, int] = {p: max(v) for p, v in pmid_positions.items()}

    heuristic_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in staging_rows:
        oa_id = (row.get("author_oa_id") or "").strip()
        orcid = (row.get("author_orcid") or "").strip()
        if oa_id or orcid:
            continue
        last = _last_name_token(row.get("author_display_name") or "")
        initial = first_initial(row.get("author_display_name") or "")
        heuristic_groups[(last, initial)].append(row)

    group_signal: dict[tuple[str, str], str] = {}
    for key, rows in heuristic_groups.items():
        pooled: list[dict] = []
        for r in rows:
            insts = r.get("institutions") or []
            if isinstance(insts, list):
                pooled.extend(i for i in insts if isinstance(i, dict))
        group_signal[key] = most_common_institution_signal(pooled)

    out: list[dict[str, Any]] = []
    for row in staging_rows:
        pmid = row.get("pmid") or ""
        position = int(row.get("author_position") or 0)
        label = (row.get("author_position_label") or "").strip()
        oa_id = (row.get("author_oa_id") or "").strip()
        orcid = (row.get("author_orcid") or "").strip()
        display_name = row.get("author_display_name") or ""

        if orcid:
            canonical = f"ORCID:{orcid}"
            method = "orcid"
        elif oa_id:
            canonical = f"OA:{oa_id}"
            method = "openalex"
        else:
            last = _last_name_token(display_name)
            initial = first_initial(display_name)
            signal = group_signal.get((last, initial), "")
            canonical = f"H:{heuristic_id(cfg.heuristic_salt, last, initial, signal)}"
            method = "heuristic"

        min_pos = pmid_min.get(pmid, position)
        max_pos = pmid_max.get(pmid, position)
        is_first = _is_first_label(label, position, min_pos)
        is_last = _is_last_label(label, position, max_pos)

        out.append(
            {
                "pmid": pmid,
                "author_position": position,
                "author_position_label": label,
                "author_oa_id": oa_id,
                "author_orcid": orcid,
                "author_display_name": display_name,
                "author_canonical_id": canonical,
                "disambiguation_method": method,
                "is_first": is_first,
                "is_last": is_last,
            }
        )
    return out
