# Build the CBT 2026 deck.
#
#   make                  build every *.tex in this folder
#   make pricing-the-defi-tail-cbt2026
#   make clean            remove LaTeX build intermediates
#   make distclean        also remove the compiled PDFs
#
# Uses latexmk when available, otherwise two pdflatex passes (the title
# page positions its parts with TikZ `remember picture`, which needs the
# second pass).

LATEXMK := $(shell command -v latexmk 2>/dev/null)
PDFLATEX := pdflatex
LFLAGS   := -pdf -interaction=nonstopmode -halt-on-error -file-line-error
PFLAGS   := -interaction=nonstopmode -halt-on-error -file-line-error

SOURCES := $(wildcard *.tex)
DECKS   := $(filter-out presentation-info,$(basename $(SOURCES)))

.PHONY: all clean distclean $(DECKS)

all: $(DECKS)

$(DECKS):
	@echo "== building $@ =="
ifdef LATEXMK
	@$(LATEXMK) $(LFLAGS) $@.tex
else
	@$(PDFLATEX) $(PFLAGS) $@.tex >/dev/null
	@$(PDFLATEX) $(PFLAGS) $@.tex >/dev/null
	@echo "   -> $@.pdf"
endif

clean:
	@rm -f *.aux *.log *.nav *.snm *.toc *.out *.vrb *.fls *.fdb_latexmk *.synctex.gz
	@echo "cleaned LaTeX build intermediates"

distclean: clean
	@rm -f $(addsuffix .pdf,$(DECKS))
	@echo "removed compiled PDFs"
