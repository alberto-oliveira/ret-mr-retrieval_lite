#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import glob
import os

from sklearn.metrics import precision_score

import numpy as np
import ipdb as pdb

completedir = lambda d: d if d[-1] == "/" else d + "/"

rk_dtype = dict(names=('name', 'votes', 'normv', 'dists', 'normd'),
                formats=('U100', np.float32, np.float32, np.float32, np.float32))

def read_rank(fpath):

    arr = np.loadtxt(fpath, dtype=rk_dtype)

    return arr

def get_label(basename):

    parts = basename.rsplit("_", 1)
    return parts[0]

def get_query_label(q_basename):

    parts = q_basename.split("_", 1)
    #parts = parts[1].rsplit("_", 1)
    return get_label(parts[1])

def get_rank_relevance(qlabel, rank):

    rksz = rank['name'].shape[0]

    gt = np.zeros((1, rksz), dtype=np.uint8)

    for i in range(rksz):
        rlabel = get_label(rank['name'][i])

        #rlabel = rlabel.split('\'')[1]
        if rlabel == qlabel:
            gt[0, i] = 1
        else:
            gt[0, i] = 0

    return gt

def evaluate_and_label(retcfg, lbl_suffix=""):

    kp = retcfg.getint("eval", "k")

    rkdir = completedir(retcfg['path']['outdir']) + "queryfiles/"
    outdir = completedir(retcfg['path']['outdir'])

    print(rkdir)

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

    for k in [1, 3, 5, 10, 25, 50, 100, 250, 500, 1000]:
        aux.append(np.mean(gtarr[:, 0:k], axis=1).reshape((-1,1)))

    prectable = np.hstack(aux)

    # Computing and stacking the average between all queries
    prectable = np.vstack([prectable, np.mean(prectable, axis=0)])

    outrelfname = outdir + retcfg['DEFAULT']['expname'] + lbl_suffix + ".irp_lbls.npy"
    outevalfname = outdir + retcfg['DEFAULT']['expname'] + "_eval.csv"

    np.save(outrelfname, gtarr[:, 0:kp])
    np.savetxt(outevalfname, prectable, header=",P@001,P@003,P@005,P@010,P@025,P@050,P@100,P@250,P@500,P@1000",
               fmt="  ,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f", delimiter=",")
