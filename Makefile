.PHONY: reproduce-paper paper-pdf

reproduce-paper:
	python3 scripts/reproduce_paper.py

paper-pdf:
	cd paper && pdflatex -interaction=nonstopmode hyperkan_verified_symbolic_rewrite_search.tex
