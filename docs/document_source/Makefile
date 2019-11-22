# 
# Adapted from: https://drewsilcock.co.uk/using-make-and-latexmk  
#

LATEX=pdflatex
LATEXOPT=--shell-escape
NONSTOP=--interaction=nonstopmode
BIBTEX=bibtex

LATEXMK=latexmk
LATEXMKOPT=-pdf
CONTINUOUS=-pvc

MAIN=master
SOURCES=$(MAIN).tex Makefile
FIGURES := $(shell find figs/* -type f)
REFERENCES := $(shell find *.bib -type f)

all: $(MAIN).pdf

.refresh:
	touch .refresh

$(MAIN).pdf: $(MAIN).tex $(MAIN).bbl .refresh $(SOURCES) $(FIGURES)
	$(LATEXMK) $(LATEXMKOPT) -pdflatex="$(LATEX) $(LATEXOPT) $(NONSTOP) %O %S" $(MAIN)

$(MAIN).bbl: $(MAIN).tex $(REFERENCES)
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)

clean:
	$(LATEXMK) -C $(MAIN)
	rm -f $(MAIN).pdfsync
	rm -rf *~ *.tmp
	rm -f *.bbl *.blg *.brf *.aux *.end *.fls *.log *.out *.fdb_latexmk
	rm -f .refresh
	rm -f *.run.xml
	rm -f *.thm
	rm -f *-blx.bib

debug:
	$(PKGS) $(LATEX) $(LATEXOPT) $(MAIN)

.PHONY: clean all
