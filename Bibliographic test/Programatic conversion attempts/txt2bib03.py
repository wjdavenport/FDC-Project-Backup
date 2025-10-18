#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
txt2bib03.py — Convert a plain-text bibliography into BibTeX (heuristic, delimiter-aware).

Usage:
  python3 txt2bib03.py input.txt -o output.bib --csv audit.csv --delimiter "===REF==="
Options:
  --per-line     Treat each physical line as a record (ONLY if you truly have 1 ref per line).
  --delimiter    A unique line token that separates records (recommended).
  --encoding     Input encoding (default: utf-8).

Notes:
- Delimiter is safest: put a line containing only your token (e.g., ===REF===) between entries.
- Output is UTF-8; ideal for biblatex/biber. Classic BibTeX may need diacritics escaped manually.
"""

import re
import argparse
import unicodedata
from collections import Counter
from pathlib import Path

# ---------- Regexes & cues ----------
RE_AUTHORISH = re.compile(r"^[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.\- ]+,")
RE_YEAR_PARENS = re.compile(r"\((\d{4})\)")
RE_YEAR_ANY    = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
RE_PAGES       = re.compile(r"(\d{1,5}[\-–]\d{1,5}|e\d{5,}|R\d{3}\-R\d{3})")
RE_URL         = re.compile(r"(https?://\S+|www\.\S+)")
RE_DOI         = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)")
# (Edition, City: Publisher, Year) — edition is optional
RE_BOOK_CIT = re.compile(
    r"\((?:(?P<edition>[^:,]+)\s*,\s*)?(?P<city>[^:]+):\s*(?P<publisher>[^,]+),\s*(?P<year>\d{4})\)"
)
RE_QUOTED_TIT  = re.compile(r"[‘“\"]([^’”\"]+)[’”\"]")
RE_BOOKTITLE_CHAP = re.compile(r",\s*in\s+(.+?)(?:,\s*ed\. by\b|\s*\()", re.IGNORECASE)

JOURNAL_CUES = [
    "Journal of", "Cognitive Neuroscience", "Current Opinion in Neurobiology",
    "Trends in Cognitive Sciences", "Trends in Neurosciences", "Frontiers in Psychology",
    "Progress in Brain Research", "Nature Neuroscience", "Nature Reviews",
    "Brain Research", "Current Biology", "Behavioral and Brain Sciences",
    "The Philosophical Review", "Philosophical Transactions", "Philosophical Quarterly",
    "Quarterly Review of Biology", "Quarterly Journal", "British Journal of Neurosurgery",
    "Neuropsychoanalysis", "PLOS Computational Biology", "PLOS ONE",
    "Epilepsy and Behavior", "Annals of", "PNAS", "Neuroimage"
]

WEB_CUES = ["stanford encyclopedia of philosophy", "internet encyclopedia of philosophy",
            "retrieved", "http://", "https://", "arxiv", "conference publication"]

PUNCT_END = re.compile(r"[.?!]\s*$")

# ---------- helpers ----------
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")

def latex_escape_min(s: str) -> str:
    return (s.replace("\\", "\\\\")
             .replace("&", "\\&").replace("%", "\\%")
             .replace("#", "\\#").replace("_", "\\_").replace("$", "\\$"))

def looks_like_journal(text: str) -> bool:
    return any(cue in text for cue in JOURNAL_CUES)

def looks_like_web(text: str) -> bool:
    low = text.lower()
    return any(c in low for c in WEB_CUES)

def classify(rec: str) -> str:
    low = rec.lower()
    if RE_BOOK_CIT.search(rec):
        if " in " in low and "ed. by" in low:
            return "incollection"
        return "book"
    if looks_like_web(rec):
        return "misc"
    if looks_like_journal(rec):
        return "article"
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

def extract_book_fields(rec: str):
    m = RE_BOOK_CIT.search(rec)
    if m:
        edition = (m.group("edition") or "").strip()
        city     = m.group("city").strip()
        pub      = m.group("publisher").strip()
        yr_in    = m.group("year")
        return pub, city, edition, yr_in
    return "", "", "", ""

def extract_booktitle_editors(rec: str):
    # Keep bracketed note immediately before ", in "
    note = ""
    mnote = re.search(r"\[([^\]]+)\]\s*,\s*in\s", rec, flags=re.IGNORECASE)
    if mnote:
        note = mnote.group(1).strip()

    m = RE_BOOKTITLE_CHAP.search(rec)
    bt = m.group(1).strip().rstrip(",") if m else ""

    e = re.search(r"ed\. by\s+([^,(]+)", rec, flags=re.IGNORECASE)
    eds = e.group(1).strip().rstrip(",;") if e else ""
    return bt, eds, note

def extract_author_title(rec: str, etype: str):
    # Prefer quoted titles for articles/chapters/web
    m = RE_QUOTED_TIT.search(rec)
    if m and etype in ("article", "incollection", "misc"):
        title = m.group(1).strip()
        authors = rec[:m.start()].strip().rstrip(",")
        return authors, title

    # Books & chapters: split at last comma BEFORE (City: Publisher, Year)
    if etype in ("book", "incollection"):
        mcit = RE_BOOK_CIT.search(rec)
        if mcit:
            start_paren = mcit.start()
            prefix = rec[:start_paren].rstrip()
            last_comma = prefix.rfind(",")
            if last_comma != -1:
                authors = prefix[:last_comma].strip()
                title   = prefix[last_comma+1:].strip().lstrip("‘“\"").rstrip(",")
                return authors, title
            return "", prefix.strip().rstrip(",")

    # Fallback: author up to first comma; title until next delimiter
    if RE_AUTHORISH.match(rec):
        first_comma = rec.find(",")
        authors = rec[:first_comma].strip()
        rest = rec[first_comma+1:].lstrip()
        stop = len(rest)
        for token in [" (", ": ", "; ", " ed. by"]:
            p = rest.find(token)
            if p != -1 and p < stop:
                stop = p
        title = rest[:stop].strip().strip(",")
        return authors, title

    return "", rec.strip()

def extract_journal_and_volume(rec: str, year: str):
    m = re.search(r",\s*([^,]+?)\s+(\d{1,4})\s*\(" + re.escape(year) + r"\)", rec) if year else None
    if m:
        return m.group(1).strip(), m.group(2).strip()
    for cue in JOURNAL_CUES:
        if cue in rec:
            return cue, ""
    return "", ""

def make_key(authors: str, year: str, title: str, used: Counter) -> str:
    last = "Item"
    if authors:
        last = re.sub(r"[^A-Za-z0-9]+", "", authors.split(",")[0].split()[-1]) or "Item"
    y = year or "n.d."
    t = re.sub(r"[^A-Za-z0-9]+", "", (title.split()[0] if title else "Title")) or "Title"
    base = f"{last}{y}{t}"
    if used[base] == 0:
        used[base] += 1
        return base
    idx = used[base]
    used[base] += 1
    return f"{base}{chr(ord('a') + idx - 1)}"

# ---------- normalization ----------
def clean_record_text(s: str) -> str:
    # Remove stray spaces after hyphen caused by line-wrap joins (e.g., "split- brain" -> "split-brain")
    s = re.sub(r"-\s+", "-", s)
    # Collapse multiple spaces
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

# ---------- segmentation ----------
def segment_with_delimiter(text: str, token: str) -> list[str]:
    # Split when a line contains exactly the token
    chunks = re.split(rf"(?m)^\s*{re.escape(token)}\s*$", text)
    return [c.strip() for c in chunks if c.strip()]

def segment_per_line(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

def segment_auto(text: str) -> list[str]:
    # Fallback auto-merger; safe but not perfect
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    merged = []
    buf = ""
    for ln in lines:
        s = ln.strip()
        if not buf:
            buf = s
            continue
        unfinished = not PUNCT_END.search(buf)
        continuation = re.match(r"^[a-z\)\],:;]", s) is not None
        if unfinished or continuation:
            buf += " " + s
        else:
            merged.append(buf)
            buf = s
    if buf:
        merged.append(buf)
    records = []
    for item in merged:
        parts = re.split(r"\.\s+(?=[A-Z])", item)
        if len(parts) == 1:
            records.append(item.strip()); continue
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

# ---------- build ----------
def to_bib_entry(rec: str, used: Counter) -> str:
    rec = clean_record_text(rec)
    etype = classify(rec)

    year  = extract_year(rec)
    url   = extract_url(rec)
    doi   = extract_doi(rec)
    pages = extract_pages(rec)

    authors, title = extract_author_title(rec, etype)
    key = make_key(authors, year, title, used)

    fields = []
    def add(k, v):
        if v:
            fields.append(f"  {k} = {{{latex_escape_min(nfc(v))}}}")

    add("author", authors)
    add("title", title)
    add("year", year)

    if etype == "article":
        journal, vol = extract_journal_and_volume(rec, year)
        add("journal", journal); add("volume", vol); add("pages", pages)
        add("doi", doi); add("url", url)

    elif etype == "book":
        publisher, address, edition, _ = extract_book_fields(rec)
        _, eds, _ = extract_booktitle_editors(rec)  # just in case it's an edited book
        add("editor", eds)
        add("edition", edition)
        add("publisher", publisher); add("address", address)
        add("doi", doi); add("url", url)

    elif etype == "incollection":
        bt, eds, note_from_brackets = extract_booktitle_editors(rec)
        publisher, address, edition, _ = extract_book_fields(rec)
        add("booktitle", bt); add("editor", eds)
        add("edition", edition)
        add("publisher", publisher); add("address", address)
        add("pages", pages); add("doi", doi); add("url", url)
        add("note", note_from_brackets)

    else:
        add("howpublished", f"\\url{{{url}}}" if url else "")
        add("doi", doi); add("url", url)

    return f"@{etype}{{{key},\n" + ",\n".join(fields) + "\n}\n"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Convert a plain-text bibliography into BibTeX.")
    ap.add_argument("input", help="Path to input .txt bibliography")
    ap.add_argument("-o", "--output", default="output.bib", help="Path to output .bib")
    ap.add_argument("--csv", help="Optional: write a CSV listing segmented references")
    ap.add_argument("--per-line", action="store_true", help="Assume one reference per line")
    ap.add_argument("--delimiter", help="Unique line token used to separate records (e.g., ===REF===)")
    ap.add_argument("--encoding", default="utf-8", help="Input file encoding (default: utf-8)")
    args = ap.parse_args()

    raw = Path(args.input).read_text(encoding=args.encoding, errors="ignore")
    text = nfc(raw)

    if args.delimiter:
        records = segment_with_delimiter(text, args.delimiter)
    elif args.per_line:
        records = segment_per_line(text)
    else:
        records = segment_auto(text)

    used = Counter()
    entries = [to_bib_entry(r, used) for r in records]

    Path(args.output).write_text("\n".join(entries), encoding="utf-8")
    print(f"Wrote {len(entries)} BibTeX entries -> {args.output}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["reference"])
            for r in records: w.writerow([r])

if __name__ == "__main__":
    main()

