#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import argparse

from libretrieval.utility import cfgloader
from create_index import create_index
from search_index import search_index
from create_ranks import create_rank_files
from evaluate_and_label import evaluate_and_label

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("cfgfile", help="Path to retrieval config file.", type=str)

    args = parser.parse_args()

    retcfg = cfgloader(args.cfgfile)

    create_index(retcfg)
    search_index(retcfg)
    create_rank_files(retcfg)
    evaluate_and_label(retcfg)