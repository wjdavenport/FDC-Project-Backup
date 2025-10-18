#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
txt2bib.py  — Convert a plain-text bibliography into BibTeX (heuristic, but tuned for philosophy/neuro refs).
Works well for:
- Books: (City: Publisher, Year) pattern
- Journal articles: "Journal Name Volume (Year), Pages"   or   "... (Year): Pages"
- Chapters in edited volumes: ... 'Title', in Book Title, ed. by Editor(s) (City: Publisher, Year), Pages
- Encyclopedia/Web items: Stanford/IEP/URLs/retrieved/arXiv/etc.  → @misc

Usage:
  python3 txt2bib.py input.txt -o output.bib --csv summary.csv
Options:
  --per-line   Assume exactly one reference per line (if your file is CR-delimited with no wrapping)
"""

import re
import argparse
import unicodedata
from collections import Counter
from pathlib import Path

# ---- Regex helpers ----
RE_AUTHORISH = re.compile(r"^[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.\- ]+,")  # "Surname,"
RE_YEAR_PARENS = re.compile(r"\((\d{4})\)")
RE_YEAR_ANY    = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
RE_PAGES       = re.compile(r"(\d{1,5}[\-–]\d{1,5}|e\d{5,}|R\d{3}\-R\d{3})")
RE_URL         = re.compile(r"(https?://\S+|www\.\S+)")
RE_DOI         = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)")
RE_EDITORS     = re.compile(r"\bed\. by ([^()]+?)(?=(\(|;|,|$))", re.IGNORECASE)
RE_BOOK_CIT    = re.compile(r"\(([^:]+):\s*([^,]+),\s*(\d{4})\)")  # (City: Publisher, Year)
RE_BOOKTITLE   = re.compile(r"\bin\s+([^(\.]+)", re.IGNORECASE)
RE_QUOTED_TIT  = re.compile(r"[‘“\"]([^’”\"]+)[’”\"]")

JOURNAL_CUES = [
    "Journal", "Neuroscience", "Neurobiology", "Neuroimage",
    "Nature", "Science", "PNAS", "Cognitive", "Quarterly",
    "Proceedings", "Annals", "Philosophical", "Brain Research",
    "Trends in", "Frontiers", "Current Biology",
    "Behavioral and Brain Sciences", "The Philosophical Review",
    "Philosophical Transactions"
]

WEB_CUES = ["stanford encyclopedia of philosophy", "internet encyclopedia of philosophy",
            "retrieved", "http://", "https://", "arxiv", "conference publication"]

PUNCT_END = re.compile(r"[.?!]\s*$")

# ---- Utilities ----
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")

def latex_escape_min(s: str) -> str:
    # Keep UTF-8 for biblatex/biber; only escape TeX specials minimally if present
    return (s.replace("\\", "\\\\")
             .replace("&", "\\&")
             .replace("%", "\\%")
             .replace("#", "\\#")
             .replace("_", "\\_")
             .replace("$", "\\$")
             )

def looks_like_journal(text: str) -> bool:
    return any(cue in text for cue in JOURNAL_CUES)

def looks_like_web(text: str) -> bool:
    low = text.lower()
    return any(c in low for c in WEB_CUES)

def classify(rec: str) -> str:
    low = rec.lower()
    if looks_like_web(rec):
        return "misc"
    if looks_like_journal(rec):
        return "article"
    # Edited/chapter vs book
    if ("(" in rec and ":" in rec and ("press" in low or "university" in low)) or " ed." in low or " ed. by" in low:
        if " in " in low and "ed. by" in low:
            return "incollection"
        return "book"
    return "misc"

def extract_year(rec: str) -> str:
    m = RE_YEAR_PARENS.search(rec)
    if m:
        return m.group(1)
    m2 = RE_YEAR_ANY.search(rec)
    return m2.group(1) if m2 else ""

def extract_pages(rec: str) -> str:
    m = RE_PAGES.search(rec)
    return m.group(1) if m else ""

def extract_url(rec: str) -> str:
    m = RE_URL.search(rec)
    return m.group(1) if m else ""

def extract_doi(rec: str) -> str:
    m = RE_DOI.search(rec)
    return m.group(1).rstrip(").,;]") if m else ""

def extract_authors(rec: str) -> str:
    # Leading comma-separated names until title/year/publisher cues
    if RE_AUTHORISH.match(rec):
        cutoff = len(rec)
        for token in ["‘", "“", " (", " in ", " ed. by", ": "]:
            p = rec.find(token)
            if p != -1 and p < cutoff:
                cutoff = p
        return rec[:cutoff].strip().rstrip(",")
    return ""

def extract_title(rec: str, authors: str) -> str:
    # Prefer quoted title
    m = RE_QUOTED_TIT.search(rec)
    if m:
        return m.group(1).strip()
    # Else between authors and first "(" or ":"/";"
    after = rec[len(authors):].lstrip(", .")
    stop = len(after)
    for token in [" (", ": ", "; ", " ed. by"]:
        p = after.find(token)
        if p != -1 and p < stop:
            stop = p
    piece = after[:stop].strip()
    if piece.lower().startswith("in "):
        piece = piece[3:].strip()
    return piece

def extract_journal_and_volume(rec: str, year: str) -> tuple[str, str]:
    """
    Try to capture '..., JournalName Volume (Year), Pages' OR ': Pages'
    """
    # Pattern with volume before (Year)
    m = re.search(r",\s*([^,]+?)\s+(\d{1,4})\s*\(" + re.escape(year) + r"\)", rec) if year else None
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Fallback: look for first journal cue as journal name
    for cue in JOURNAL_CUES:
        if cue in rec:
            return cue, ""
    return "", ""

def extract_book_fields(rec: str) -> tuple[str,str]:
    # (City: Publisher, Year)
    m = RE_BOOK_CIT.search(rec)
    if m:
        city = m.group(1).strip()
        pub  = m.group(2).strip()
        return pub, city
    return "", ""

def extract_booktitle_editors(rec: str) -> tuple[str, str]:
    # in BookTitle, ed. by Editors ...
    bt = ""
    eds = ""
    m = RE_BOOKTITLE.search(rec)
    if m:
        bt = m.group(1).strip().rstrip(",")
    e = RE_EDITORS.search(rec)
    if e:
        eds = e.group(1).strip().rstrip(",;")
    return bt, eds

def make_key(authors: str, year: str, title: str, used_keys: Counter) -> str:
    last = "Item"
    if authors:
        last = re.sub(r"[^A-Za-z0-9]+", "", authors.split(",")[0].split()[-1]) or "Item"
    y = year or "n.d."
    t = re.sub(r"[^A-Za-z0-9]+", "", (title.split()[0] if title else "Title")) or "Title"
    base = f"{last}{y}{t}"
    if used_keys[base] == 0:
        used_keys[base] += 1
        return base
    # add letter suffixes a,b,c...
    idx = used_keys[base]
    used_keys[base] += 1
    return f"{base}{chr(ord('a') + idx - 1)}"

def to_bib_entry(rec: str, used: Counter) -> str:
    rec = rec.strip()
    etype = classify(rec)
    year  = extract_year(rec)
    url   = extract_url(rec)
    doi   = extract_doi(rec)
    pages = extract_pages(rec)

    authors = extract_authors(rec)
    title   = extract_title(rec, authors)
    key     = make_key(authors, year, title, used)

    fields = []
    def add(k, v):
        if v:
            fields.append(f"  {k} = {{{latex_escape_min(nfc(v))}}}")

    add("author", authors)
    add("title", title)
    add("year", year)

    if etype == "article":
        journal, vol = extract_journal_and_volume(rec, year)
        add("journal", journal)
        add("volume", vol)
        add("pages", pages)
        add("doi", doi)
        add("url", url)
    elif etype == "book":
        publisher, address = extract_book_fields(rec)
        add("publisher", publisher)
        add("address", address)
        add("doi", doi)
        add("url", url)
    elif etype == "incollection":
        bt, eds = extract_booktitle_editors(rec)
        publisher, address = extract_book_fields(rec)
        add("booktitle", bt)
        add("editor", eds)
        add("publisher", publisher)
        add("address", address)
        add("pages", pages)
        add("doi", doi)
        add("url", url)
    else:
        # misc / web
        add("howpublished", f"\\url{{{url}}}" if url else "")
        add("doi", doi)
        add("url", url)

    return f"@{etype}{{{key},\n" + ",\n".join(fields) + "\n}\n"

# ---- Segmentation ----
def segment_records_auto(text: str) -> list[str]:
    """
    Attempts to undo soft wraps and split into references.
    Works when entries are multi-line and not separated by blank lines.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    # merge soft wraps
    merged = []
    buf = ""
    for ln in lines:
        s = ln.strip()
        if not buf:
            buf = s
            continue
        # If buffer doesn't end like a finished reference, continue merging
        unfinished = not PUNCT_END.search(buf)
        continuation = re.match(r"^[a-z\)\],:;]", s) is not None
        if unfinished or continuation:
            buf += " " + s
        else:
            merged.append(buf)
            buf = s
    if buf:
        merged.append(buf)

    # sometimes two references end up merged; try to split on ". " followed by capital+year later
    records: list[str] = []
    for item in merged:
        parts = re.split(r"\.\s+(?=[A-Z])", item)
        if len(parts) == 1:
            records.append(item.strip())
            continue
        # rebuild conservatively
        acc = parts[0]
        for p in parts[1:]:
            has_year = bool(RE_YEAR_ANY.search(acc))
            starts_authorish = bool(RE_AUTHORISH.match(p))
            if has_year and starts_authorish:
                records.append(acc.strip() + ".")
                acc = p
            else:
                acc += ". " + p
        records.append(acc.strip())
    return [r.strip() for r in records if r.strip()]

def segment_records_per_line(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

# ---- Main ----
def main():
    ap = argparse.ArgumentParser(description="Convert a plain-text bibliography into BibTeX.")
    ap.add_argument("input", help="Path to input .txt bibliography")
    ap.add_argument("-o", "--output", default="output.bib", help="Path to output .bib")
    ap.add_argument("--csv", help="Optional: write a CSV listing segmented references")
    ap.add_argument("--per-line", action="store_true",
                    help="Assume one reference per line (no soft wrap merging)")
    args = ap.parse_args()

    raw = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    text = nfc(raw)

    records = segment_records_per_line(text) if args.per_line else segment_records_auto(text)

    used = Counter()
    entries = [to_bib_entry(r, used) for r in records]

    Path(args.output).write_text("\n".join(entries), encoding="utf-8")
    print(f"Wrote {len(entries)} BibTeX entries -> {args.output}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["reference"])
            for r in records:
                w.writerow([r])

if __name__ == "__main__":
    main()

