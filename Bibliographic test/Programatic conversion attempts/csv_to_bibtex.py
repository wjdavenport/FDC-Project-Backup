import csv, sys, re

def esc(s):
    if s is None: return ""
    # minimal brace escaping
    return s.replace("{", "").replace("}", "")

def main(csv_path, out_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        entrytype = (r.get("entrytype") or "misc").strip()
        key = (r.get("key") or "item").strip()
        if not key:
            # derive a crude key from author+year+first word of title
            a = (r.get("author") or "anon").split(",")[0].split()[-1]
            y = (r.get("year") or "n.d.")
            t = (r.get("title") or "untitled").split()[0]
            key = f"{a}{y}{t}"
        fields = []
        for k,v in r.items():
            if k in ("entrytype","key") or not v.strip():
                continue
            fields.append(f"  {k} = {{{esc(v.strip())}}}")
        bib = f"@{entrytype}{{{key},\n" + ",\n".join(fields) + "\n}\n\n"
        out.append(bib)
    with open(out_path, "w", encoding="utf-8") as g:
        g.write("".join(out))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 csv_to_bibtex.py input.csv output.bib")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])