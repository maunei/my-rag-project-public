"""
NCBI E-utilities (PubMed) — esearch + esummary + efetch for abstract_meta.

The **NCBI abstract text** is returned separately from ``pubmed_enrichment`` so each JSON
record stays lean: ``abstract_text`` (PDF) and ``abstract_pubmed`` (NCBI) only at the
top level; ``pubmed_enrichment`` holds PMID metadata and ``esummary`` only.

Reads **NCBI credentials** from the environment variables ``NCBI_EMAIL`` /
``NCBI_API_KEY`` (or ``ENTREZ_EMAIL`` / ``ENTREZ_API_KEY``): contact email plus
optional API key — typically supplied via ``.env`` loaded by ``extract_abstracts.py``.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

EUTIL_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_ENRICHMENT_SCHEMA = 3
REQUEST_TIMEOUT = 45


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sleep_for_rate_limit(has_api_key: bool) -> None:
    """Space requests to stay within NCBI limits (≈10/s with key, ≈3/s without)."""
    time.sleep(0.11 if has_api_key else 0.35)


def _common_params(email: str | None, api_key: str | None) -> dict[str, str]:
    p: dict[str, str] = {"tool": "papers_rag_extract_abstracts", "retmode": "json"}
    if email:
        p["email"] = email
    if api_key:
        p["api_key"] = api_key
    return p


def _efetch_params(email: str | None, api_key: str | None) -> dict[str, str]:
    """Efetch often uses ``retmode=xml``; do not force JSON retmode here."""
    p: dict[str, str] = {"tool": "papers_rag_extract_abstracts"}
    if email:
        p["email"] = email
    if api_key:
        p["api_key"] = api_key
    return p


def abstract_text_from_pubmed_efetch_xml(xml_text: str) -> str:
    """
    Parse PubMed ``efetch`` XML (``rettype=abstract``) and join ``AbstractText`` nodes.
    Structured abstracts use ``Label`` attributes (e.g. BACKGROUND: ...).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""

    chunks: list[str] = []
    for node in root.findall(".//AbstractText"):
        label = (node.get("Label") or "").strip()
        body = "".join(node.itertext()).strip()
        body = " ".join(body.split())
        if not body:
            continue
        if label:
            chunks.append(f"{label}: {body}")
        else:
            chunks.append(body)
    return "\n\n".join(chunks).strip()


def fetch_abstract_via_efetch(
    pmid: str,
    *,
    email: str | None,
    api_key: str | None,
) -> tuple[str, str | None]:
    """
    Fetch MedlineCitation abstract via ``efetch``.

    Returns (abstract_text, error_or_none).
    """
    params = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "abstract",
        "retmode": "xml",
        **_efetch_params(email, api_key),
    }
    try:
        r = requests.get(
            f"{EUTIL_BASE}/efetch.fcgi",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        text = abstract_text_from_pubmed_efetch_xml(r.text)
        return text, None
    except requests.RequestException as exc:
        return "", f"efetch:{exc}"


def _sanitize_title_for_term(title: str) -> str:
    t = title.strip().replace('"', " ").replace("\n", " ").replace("[", " ").replace("]", " ")
    while "  " in t:
        t = t.replace("  ", " ")
    return t[:500]


def _build_title_term(title: str) -> str:
    """PubMed field tag search on the cleaned title text (matches Entrez `[Title]` behavior)."""
    clean = _sanitize_title_for_term(title)
    if not clean:
        return ""
    # Do not wrap the title in inner double-quotes — that often yields zero hits versus the same phrase without quotes.
    return f"{clean}[Title]"


def sanitize_doi_for_pubmed(raw: str) -> str:
    """Strip URL / ``doi:`` prefix for PubMed `[doi]` term."""
    if not raw or not str(raw).strip():
        return ""
    d = str(raw).strip()
    d = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", d, flags=re.I)
    d = re.sub(r"(?i)^doi:\s*", "", d).strip()
    t = d
    while t and t[-1] in ".,;:!?)]}\"'›»":
        t = t[:-1]
    return t.strip()


def _fresh_enrichment(status: str) -> dict:
    return {
        "schema_version": PUBMED_ENRICHMENT_SCHEMA,
        "queried_at": _now_iso(),
        "lookup_route": None,
        "search_query": "",
        "doi_normalized": None,
        "status": status,
        "pmid": None,
        "candidate_pmids": None,
        "disambiguation_note": None,
        "error": None,
        "esummary": None,
    }


def _esearch_idlist_and_detail(data: dict) -> tuple[list[str], str | None]:
    """
    Parse ``esearch`` JSON ``esearchresult`` into PMID id strings.

    NCBI sometimes returns ``{"esearchresult": {"ERROR": "..."}}`` with **no** ``idlist`` key
    during backend faults — callers must surface that instead of treating it as generic shape error.

    Returns ``(ids, detail)`` where **detail** means “do not proceed to esummary”:
    - ``None`` — ok to use ``ids`` (may be empty → no PMID hit)
    - ``esearch_ncbi:…`` — NCBI ``ERROR`` field (transient or query failure)
    - ``esearch_shape:…`` — unexpected JSON envelope
    """
    sr = data.get("esearchresult")
    if not isinstance(sr, dict):
        return [], "esearch_shape:esearchresult_not_object"

    err_raw = sr.get("ERROR")
    if err_raw is not None:
        msg = str(err_raw).strip()
        if msg:
            cap = msg[:900] + ("…" if len(msg) > 900 else "")
            return [], f"esearch_ncbi:{cap}"

    raw_ids = sr.get("idlist")
    if raw_ids is None:
        return [], None
    if not isinstance(raw_ids, list):
        return [], f"esearch_shape:idlist_not_list:{type(raw_ids).__name__}"

    cleaned = [str(x).strip() for x in raw_ids if str(x).strip()]
    return cleaned, None


def _esearch_json_with_retry(params: dict, *, has_api_key: bool) -> tuple[dict | None, str | None]:
    """
    Fetch ``esearch.fcgi``. If Entrez responds with ``esearchresult.ERROR``, wait briefly and retry once.

    Returns ``(parsed_json_or_none, error_string_or_none)`` for HTTP-level / JSON-parse failures only.
    """
    for attempt in (0, 1):
        try:
            r = requests.get(
                f"{EUTIL_BASE}/esearch.fcgi",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as exc:
            return None, f"esearch:{exc}"
        except ValueError as exc:
            return None, f"esearch_json:{exc}"

        _, detail = _esearch_idlist_and_detail(data)
        if detail and detail.startswith("esearch_ncbi:") and attempt == 0:
            time.sleep(0.85 if has_api_key else 1.85)
            continue
        return data, None

    return data, None


def fetch_pubmed_enrichment_for_title(
    title: str,
    *,
    email: str | None,
    api_key: str | None,
) -> tuple[dict, str, str | None]:
    """
    Return ``(pubmed_enrichment_no_abstract, abstract_from_ncbi, efetch_error_or_none)``.

    The enrichment dict never contains duplicate abstract bodies (see schema_version 3).
    """
    has_key = bool(api_key and api_key.strip())

    empty = "", None
    base = _fresh_enrichment("skipped_missing_title")
    if not (title and title.strip()):
        return base, *empty

    term = _build_title_term(title)
    base["search_query"] = term
    base["lookup_route"] = "title"

    params = {
        "db": "pubmed",
        "term": term,
        "retmax": 20,
        "retstart": 0,
        "sort": "relevance",
        **_common_params(email, api_key),
    }

    data, outer_err = _esearch_json_with_retry(params, has_api_key=has_key)
    if outer_err:
        fail = (
            _fresh_enrichment("http_error")
            if outer_err.startswith("esearch:") and not outer_err.startswith("esearch_json:")
            else _fresh_enrichment("parse_error")
        )
        fail["search_query"] = term
        fail["lookup_route"] = "title"
        fail["error"] = outer_err
        return fail, *empty

    id_list, es_detail = _esearch_idlist_and_detail(data)
    if es_detail:
        fail = _fresh_enrichment(
            "ncbi_esearch_error" if es_detail.startswith("esearch_ncbi:") else "parse_error"
        )
        fail["search_query"] = term
        fail["lookup_route"] = "title"
        fail["error"] = es_detail
        return fail, *empty

    if not id_list:
        b = _fresh_enrichment("no_hit")
        b["search_query"] = term
        b["lookup_route"] = "title"
        return b, *empty

    candidate_pmids = id_list[:]
    ambiguous_multi_title = len(id_list) > 1
    pmid = id_list[0]
    base["pmid"] = pmid

    _sleep_for_rate_limit(has_key)

    sparams = {
        "db": "pubmed",
        "id": pmid,
        **_common_params(email, api_key),
    }
    try:
        r2 = requests.get(
            f"{EUTIL_BASE}/esummary.fcgi",
            params=sparams,
            timeout=REQUEST_TIMEOUT,
        )
        r2.raise_for_status()
        sdata = r2.json()
    except requests.RequestException as exc:
        b = _fresh_enrichment("http_error")
        b["lookup_route"] = "title"
        b["search_query"] = term
        b["pmid"] = pmid
        b["error"] = f"esummary:{exc}"
        return b, *empty
    except ValueError as exc:
        b = _fresh_enrichment("parse_error")
        b["lookup_route"] = "title"
        b["search_query"] = term
        b["pmid"] = pmid
        b["error"] = f"esummary_json:{exc}"
        return b, *empty

    try:
        rec = sdata["result"][pmid]
    except (KeyError, TypeError) as exc:
        b = _fresh_enrichment("parse_error")
        b["lookup_route"] = "title"
        b["search_query"] = term
        b["pmid"] = pmid
        b["error"] = f"esummary_shape:{exc}"
        return b, *empty

    if ambiguous_multi_title:
        base["status"] = "ambiguous_using_first"
        base["candidate_pmids"] = candidate_pmids
        base["disambiguation_note"] = "multiple_title_hits_used_esearch_relevance_rank_first"
    else:
        base["status"] = "ok"
        base["candidate_pmids"] = None

    base["esummary"] = rec

    _sleep_for_rate_limit(has_key)
    ab_txt, ab_err = fetch_abstract_via_efetch(pmid, email=email, api_key=api_key)
    return base, ab_txt or "", ab_err


def fetch_pubmed_enrichment_for_doi(
    doi: str,
    *,
    email: str | None,
    api_key: str | None,
) -> tuple[dict, str, str | None]:
    """
    Resolve PubMed summary + abstract using ``DOI`` field-tag search (`{doi}[doi]`).

    Return shape matches :func:`fetch_pubmed_enrichment_for_title`.
    """
    has_key = bool(api_key and api_key.strip())
    empty = "", None

    base = _fresh_enrichment("skipped_missing_doi")
    clean = sanitize_doi_for_pubmed(doi)
    base["doi_normalized"] = clean or None
    base["lookup_route"] = "doi"

    if not clean:
        return base, *empty

    if not re.match(r"(?i)10\.\d{4,9}/\S+", clean):
        b = _fresh_enrichment("skipped_invalid_doi")
        b["lookup_route"] = "doi"
        b["doi_normalized"] = clean
        b["search_query"] = f"{clean}[doi]"
        return b, *empty

    term = f"{clean}[doi]"
    base["search_query"] = term

    params = {
        "db": "pubmed",
        "term": term,
        "retmax": 10,
        "retstart": 0,
        "sort": "relevance",
        **_common_params(email, api_key),
    }

    data, outer_err = _esearch_json_with_retry(params, has_api_key=has_key)
    if outer_err:
        fail = (
            _fresh_enrichment("http_error")
            if outer_err.startswith("esearch:") and not outer_err.startswith("esearch_json:")
            else _fresh_enrichment("parse_error")
        )
        fail["lookup_route"] = "doi"
        fail["doi_normalized"] = clean
        fail["search_query"] = term
        fail["error"] = outer_err
        return fail, *empty

    id_list, es_detail = _esearch_idlist_and_detail(data)
    if es_detail:
        fail = _fresh_enrichment(
            "ncbi_esearch_error" if es_detail.startswith("esearch_ncbi:") else "parse_error"
        )
        fail["lookup_route"] = "doi"
        fail["doi_normalized"] = clean
        fail["search_query"] = term
        fail["error"] = es_detail
        return fail, *empty

    if not id_list:
        b = _fresh_enrichment("no_hit")
        b["lookup_route"] = "doi"
        b["doi_normalized"] = clean
        b["search_query"] = term
        return b, *empty

    candidate_pmids = id_list[:]
    ambiguous_multi = len(id_list) > 1
    pmid = id_list[0]
    base["pmid"] = pmid

    _sleep_for_rate_limit(has_key)

    sparams = {
        "db": "pubmed",
        "id": pmid,
        **_common_params(email, api_key),
    }
    try:
        r2 = requests.get(
            f"{EUTIL_BASE}/esummary.fcgi",
            params=sparams,
            timeout=REQUEST_TIMEOUT,
        )
        r2.raise_for_status()
        sdata = r2.json()
    except requests.RequestException as exc:
        b = _fresh_enrichment("http_error")
        b["lookup_route"] = "doi"
        b["doi_normalized"] = clean
        b["search_query"] = term
        b["pmid"] = pmid
        b["error"] = f"esummary:{exc}"
        return b, *empty
    except ValueError as exc:
        b = _fresh_enrichment("parse_error")
        b["lookup_route"] = "doi"
        b["doi_normalized"] = clean
        b["search_query"] = term
        b["pmid"] = pmid
        b["error"] = f"esummary_json:{exc}"
        return b, *empty

    try:
        rec = sdata["result"][pmid]
    except (KeyError, TypeError) as exc:
        b = _fresh_enrichment("parse_error")
        b["lookup_route"] = "doi"
        b["doi_normalized"] = clean
        b["search_query"] = term
        b["pmid"] = pmid
        b["error"] = f"esummary_shape:{exc}"
        return b, *empty

    if ambiguous_multi:
        base["status"] = "ambiguous_doi_using_first"
        base["candidate_pmids"] = candidate_pmids
        base["disambiguation_note"] = "doi_query_returned_multiple_pmids_used_esearch_rank_first"
    else:
        base["status"] = "ok"
        base["candidate_pmids"] = None

    base["doi_normalized"] = clean
    base["esummary"] = rec

    _sleep_for_rate_limit(has_key)
    ab_txt, ab_err = fetch_abstract_via_efetch(pmid, email=email, api_key=api_key)
    return base, ab_txt or "", ab_err
