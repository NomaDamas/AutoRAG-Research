# AutoRAG-Research NeurIPS 2026 Paper

This directory contains the NeurIPS 2026 Evaluations & Datasets-track LaTeX manuscript for the AutoRAG-Research paper.

Official starter kit source:
https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip

Files:
- `main.tex` — main paper source, written in final-submission style with result values represented by `XX`/`0.XXX` review markers.
- `main.pdf` — compiled PDF.
- `neurips_2026.sty` — copied from the official starter kit.
- `checklist.tex` — NeurIPS checklist answers aligned with the manuscript claims.
- `references.bib` — bibliography.
- `framework-paper-patterns.md` — saved analysis of major framework-paper structure used for the rewrite.
- `starter-kit/` — official starter-kit files kept for provenance.

Build:

```bash
cd paper/neurips2026
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
