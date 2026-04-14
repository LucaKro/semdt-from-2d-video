PAPER = main

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib IEEEtran.cls IEEEtran.bst
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log \
	      $(PAPER).out $(PAPER).toc $(PAPER).synctex.gz

distclean: clean
	rm -f $(PAPER).pdf

.PHONY: all clean distclean
