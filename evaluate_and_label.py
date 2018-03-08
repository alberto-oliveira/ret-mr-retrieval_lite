#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import glob
import os

from sklearn.metrics import precision_score

import numpy as np

sys.path.append("/home/alberto/SpotME/projects/performance-prediction/sources/ret-mr-learning/source/")
from common.rIO import read_rank

completedir = lambda d: d if d[-1] == "/" else d + "/"

def get_label(basename):

    parts = basename.rsplit("_", 2)
    return parts[0]

def get_query_label(q_basename):

    parts = q_basename.split("_", 1)
    return get_label(parts[1])

def get_rank_relevance(qlabel, rank):

    rksz = rank['name'].shape[0]

    gt = np.zeros((1, rksz), dtype=np.uint8)

    for i in range(rksz):
        rlabel = get_label(rank['name'][i])
        if rlabel == qlabel:
            gt[0, i] = 1
        else:
            gt[0, i] = 0

    return gt

def evaluate_and_label(retcfg):

    kp = retcfg.getint("eval", "k")

    rkdir = completedir(retcfg['path']['outdir']) + "queryfiles/"
    outdir = completedir(retcfg['path']['outdir'])

    rkfiles = glob.glob(rkdir + "*.rk")
    rkfiles.sort()

    gtlist = []
    for rkpath in rkfiles:
        print("Processing:", os.path.basename(rkpath))
        qlabel = get_query_label(os.path.basename(rkpath))

        rank = read_rank(rkpath)
        gtlist.append(get_rank_relevance(qlabel, rank))
        print("---")

    gtarr = np.vstack(gtlist)
    print(gtarr.shape)
    aux = []

    for k in [1, 3, 5, 10, 25, 50, 100]:
        aux.append(np.mean(gtarr[:, 0:k], axis=1).reshape((-1,1)))

    prectable = np.hstack(aux)

    # Computing and stacking the average between all queries
    prectable = np.vstack([prectable, np.mean(prectable, axis=0)])

    outrelfname = outdir + retcfg['DEFAULT']['expname'] + "_rel.out"
    outevalfname = outdir + retcfg['DEFAULT']['expname'] + "_eval.csv"

    np.save(outrelfname, gtarr[:, 0:kp])
    np.savetxt(outevalfname, prectable, header=",P@001,P@003,P@005,P@010,P@025,P@050,P@100",
               fmt="  ,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f", delimiter=",")
