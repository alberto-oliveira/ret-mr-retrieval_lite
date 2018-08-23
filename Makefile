help:
		@cat Makefile

CFGDIR?="./config/"
OUTDIR?="./output/"
QDIR?="./queries/"

index: create_index.py
		python create_index.py $(CFGPATH)

search: search_index.py
		python search_index.py $(CFGPATH)

rank: create_ranks.py
		python create_ranks.py $(CFGPATH)

evaluate: evaluate_and_label.py
		python evaluate_and_label.py $(CFGPATH)

pipeline: index search rank evaluate
		@echo "-- Pipeline for: "
		@cat $(CFGFILE)

fitting:
		python fitting.py $(CFGPATH)

.PHONY: cleanout view list

cleanout:
		rm -rfv $(OUTDIR)`sed -n -e 's/expname=//p' $(CFGPATH)`

view:
		@cat $(CFGPATH)

list:
		@ls $(CFGDIR)
