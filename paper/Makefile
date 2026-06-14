# ============================================================
# DeIR-Dual V2 论文编译 Makefile
# ============================================================

PAPER = paper
TEX_DIR = tex
TEXLIVE_BIN = /home/luwa/.texlive/bin/x86_64-linux

.PHONY: all clean cleanall pdf quick

# 完整编译（含参考文献）
all: pdf

pdf:
	cd $(TEX_DIR) && $(TEXLIVE_BIN)/pdflatex -interaction=nonstopmode $(PAPER).tex
	cd $(TEX_DIR) && $(TEXLIVE_BIN)/bibtex $(PAPER)
	cd $(TEX_DIR) && $(TEXLIVE_BIN)/pdflatex -interaction=nonstopmode $(PAPER).tex
	cd $(TEX_DIR) && $(TEXLIVE_BIN)/pdflatex -interaction=nonstopmode $(PAPER).tex
	@echo "编译完成：$(TEX_DIR)/$(PAPER).pdf"

# 快速编译（不更新参考文献）
quick:
	cd $(TEX_DIR) && $(TEXLIVE_BIN)/pdflatex -interaction=nonstopmode $(PAPER).tex
	cd $(TEX_DIR) && $(TEXLIVE_BIN)/pdflatex -interaction=nonstopmode $(PAPER).tex
	@echo "快速编译完成：$(TEX_DIR)/$(PAPER).pdf"

# 清理中间文件
clean:
	rm -f $(TEX_DIR)/$(PAPER).aux $(TEX_DIR)/$(PAPER).bbl $(TEX_DIR)/$(PAPER).blg
	rm -f $(TEX_DIR)/$(PAPER).log $(TEX_DIR)/$(PAPER).out $(TEX_DIR)/$(PAPER).toc
	rm -f $(TEX_DIR)/$(PAPER).lof $(TEX_DIR)/$(PAPER).lot

# 清理所有（含 PDF）
cleanall: clean
	rm -f $(TEX_DIR)/$(PAPER).pdf
