
TEX = $(wildcard *.tex)
BIB = $(patsubst %.tex, %, $(TEX))

default: bib pdf

bib:
	biber $(BIB)
# needs texlive
pdf:
	pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error $(TEX)
zip: 
	zip Daten.zip -R Daten/
# needs inotify-tools
job:
	while inotifywait -e close_write $(TEX) || true; do make pdf ; done
