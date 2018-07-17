help:
		@cat Makefile

CFGDIR?="./config/"
OUTDIR?="./output/
QDIR?="./queries/

index: create_index.py
		python create_index.py $(CFGDIR)$(CFGFILE)

search: search_index.py
		python search_index.py $(CFGDIR)$(CFGFILE)

rank: create_ranks.py
		python create_ranks.py $(CFGDIR)$(CFGFILE)

evaluate: evaluate_and_label.py
		python evaluate_and_label.py $(CFGDIR)$(CFGFILE)

pipeline: index search rank evaluate
		@echo "-- Pipeline for: "
		@cat $(CFGDIR)$(CFGFILE)

.PHONY: check

check:
