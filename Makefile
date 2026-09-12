PYTHON ?= python

.PHONY: eval handle agreement prepare explore

eval:
	$(PYTHON) -m src.cli eval

handle:
	$(PYTHON) -m src.cli handle "my hulu keeps buffering on roku during the game"

agreement:
	$(PYTHON) -m src.cli agreement

prepare:
	$(PYTHON) -m src.cli prepare

explore:
	$(PYTHON) -m src.cli explore
