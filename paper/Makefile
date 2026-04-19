# Paper build orchestrator.
#
# Targets
# -------
# paper    -- re-run the pipeline, stamp macros, build the PDF.
# quick    -- just stamp macros and rebuild the PDF (no pipeline).
# figures  -- copy outputs/plots/*.png into paper/figures/.
# clean    -- remove LaTeX build artefacts.
#
# The PDF build is a graceful no-op if neither latexmk nor pdflatex is on
# PATH.

.PHONY: paper quick figures clean

LATEXMK := $(shell command -v latexmk 2>/dev/null)
PDFLATEX := $(shell command -v pdflatex 2>/dev/null)

REPO_ROOT := $(realpath ..)
PLOT_SRC  := $(REPO_ROOT)/outputs/plots
FIG_DST   := figures
SUMMARY   := $(REPO_ROOT)/outputs/summary.json

paper:
	uv run python $(REPO_ROOT)/scripts/run_full_pipeline.py
	uv run python build_paper.py --summary $(SUMMARY) --tex main.tex --out main.tex
	$(MAKE) figures
	$(MAKE) quick

quick:
	uv run python build_paper.py --summary $(SUMMARY) --tex main.tex --out main.tex
ifneq ($(LATEXMK),)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex
else
ifneq ($(PDFLATEX),)
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
else
	@echo "[paper] neither latexmk nor pdflatex on PATH; skipping PDF build."
endif
endif

figures:
	mkdir -p $(FIG_DST)
	@if [ -d "$(PLOT_SRC)" ]; then \
		cp -f $(PLOT_SRC)/*.png $(FIG_DST)/ 2>/dev/null || true ; \
		echo "[paper] figures synchronised from $(PLOT_SRC)" ; \
	else \
		echo "[paper] no $(PLOT_SRC); run the pipeline first." ; \
	fi
	@if [ -f "$(REPO_ROOT)/outputs/benchmarks.png" ]; then \
		cp -f $(REPO_ROOT)/outputs/benchmarks.png $(FIG_DST)/ ; \
		echo "[paper] copied outputs/benchmarks.png -> $(FIG_DST)/" ; \
	fi

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.fdb_latexmk main.fls main.synctex.gz
	rm -f main.pdf
