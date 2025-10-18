# Minimal BibTeX Cheat Sheet (for fast manual entry)

**Keys**: Use a stable scheme like `LastnameYYYYShortTitle` (no spaces).  
**Author list**: Use `and` between authors. Example: `Chalmers, David J. and Dennett, Daniel C.`

## Common entry types

### Journal article
```bibtex
@article{Key,
  author = {Last, First and Last, First M.},
  title = {Title of the Article},
  journal = {Journal Name},
  year = {YYYY},
  volume = {VV},
  number = {II},
  pages = {pp–pp},
  doi = {10.xxxx/xxxxx},
  url = {https://...},
  note = {optional}
}
```

### Book
```bibtex
@book{Key,
  author = {Last, First},
  title = {Book Title},
  publisher = {Publisher},
  address = {City},
  year = {YYYY},
  isbn = {ISBN},
  note = {optional}
}
```

### Chapter in an edited volume
```bibtex
@incollection{Key,
  author = {Last, First},
  title = {Chapter Title},
  booktitle = {Book Title},
  editor = {EditorLast, EditorFirst},
  publisher = {Publisher},
  address = {City},
  year = {YYYY},
  pages = {pp–pp},
  doi = {10.xxxx/xxxxx}
}
```

### Conference paper
```bibtex
@inproceedings{Key,
  author = {Last, First},
  title = {Paper Title},
  booktitle = {Proceedings of ...},
  year = {YYYY},
  pages = {pp–pp},
  doi = {10.xxxx/xxxxx}
}
```

### Thesis
```bibtex
@phdthesis{Key,
  author = {Last, First},
  title = {Thesis Title},
  school = {University},
  year = {YYYY},
  address = {City},
  type = {PhD thesis},
  url = {https://...}
}
```

### Web / Online
```bibtex
@misc{Key,
  author = {Last, First},
  title = {Page or Post Title},
  year = {YYYY},
  howpublished = {\url{https://...}},
  note = {Accessed YYYY-MM-DD}
}
```

## Fast tips
- If a DOI exists, include it — it's your best dedup key.
- Use en-dash for page ranges (`123--145`) if your LaTeX toolchain expects it.
- Keep capitalization you care about with braces in the title: `{IIT}` to prevent lowercasing.
- For multiple editions or reprints, add details in `note` (e.g., `orig. 1890; edition 1918`).

## Workflow suggestion
1. Enter rows in `bibliography_manual_template.csv`.
2. Run: `python3 csv_to_bibtex.py bibliography_manual_template.csv output.bib`.
3. Import `output.bib` into Zotero (optional) or use directly for matching.