help:
		@cat Makefile

CFGDIR?="./config/"
OUTDIR?="./output/"
QDIR?="./queries/"

index: create_index.py
		python create_index.py $(CFGFILE)

search: search_index.py
		python search_index.py $(CFGFILE)

rank: create_ranks.py
		python create_ranks.py $(CFGFILE)

evaluate: evaluate_and_label.py
		python evaluate_and_label.py $(CFGFILE)

pipeline: index search rank evaluate
		@echo "-- Pipeline for: "
		@cat $(CFGFILE)

fitting:
		python fitting.py $(CFGFILE)

.PHONY: cleanout view list

cleanout:
		rm -rfv $(OUTDIR)`sed -n -e 's/expname=//p' $(CFGFILE)`

view:
		@cat $(CFGFILE)

list:
		@ls $(CFGDIR)
