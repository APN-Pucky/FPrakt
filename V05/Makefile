
TEX = $(wildcard *.tex)
BIB = $(patsubst %.tex, %, $(TEX))

default: bib pdf

bib:
	biber $(BIB)
pdf:
	pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error $(TEX)
mk:
	latexmk -pdf $(TEX)
zip: 
	zip Daten.zip -R Daten/

job-pdf:
	while inotifywait -e close_write $(TEX) || true; do make pdf && pkill -HUP mupdf ; done
job-mk:
	latexmk -pdf -pvc $(TEX) &>log.log &
