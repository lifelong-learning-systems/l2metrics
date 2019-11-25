# 
# Adapted from: https://drewsilcock.co.uk/using-make-and-latexmk  
#

LATEX=pdflatex
LATEXOPT=--shell-escape
NONSTOP=--interaction=nonstopmode

LATEXMK=latexmk
LATEXMKOPT=-pdf
CONTINUOUS=-pvc

MAIN=master
SOURCES=$(MAIN).tex Makefile
FIGURES := $(shell find sections/figs/* -type f)

all: $(MAIN).pdf

.refresh:
	touch .refresh

$(MAIN).pdf: $(MAIN).tex .refresh $(FIGURES)
	$(LATEXMK) $(LATEXMKOPT) -pdflatex="$(LATEX) $(LATEXOPT) $(NONSTOP) %O %S" $(MAIN)
	mv $(MAIN).pdf ../metrics_documentation.pdf
	make clean

clean:
	$(LATEXMK) -C $(MAIN)
	rm -f $(MAIN).pdfsync
	rm -rf *~ *.tmp
	rm -f *.blg *.brf *.aux *.end *.fls *.log *.out *.fdb_latexmk
	rm -f .refresh
	rm -f *.run.xml
	rm -f *.thm

debug:
	$(PKGS) $(LATEX) $(LATEXOPT) $(MAIN)

.PHONY: clean all
