#!/usr/bin/env python
#-*- coding: utf-8 -*-

import glob
import os
import argparse

from sklearn.metrics import precision_score

import numpy as np

import ipdb as pdb

from libretrieval.utility import cfgloader

completedir = lambda d: d if d[-1] == "/" else d + "/"

rk_dtype = dict(names=('name', 'score'),
                formats=('U100', np.float64))


def read_rank(fpath):

    arr = np.loadtxt(fpath, dtype=rk_dtype)

    return arr


def get_label(name):
    parts = name.split("_")
    i = 0

    for i in range(len(parts)):
        if parts[i].isdigit():
            break

    return "_".join(parts[:i])


def get_query_label(qname):
    suffix = qname.split("_", 1)[1]
    return get_label(suffix)


def get_rank_relevance(qlabel, rank, k=-1):

    rksz = rank['name'].shape[0]
    if k == -1:
        k = rksz

    gt = np.zeros((1, k), dtype=np.uint8)

    for i in range(k):
        rlabel = get_label(rank['name'][i])

        #rlabel = rlabel.split('\'')[1]
        if rlabel == qlabel:
            gt[0, i] = 1
        else:
            gt[0, i] = 0

    return gt


def evaluate_and_label(retcfg, lbl_suffix=""):

    eval_k = retcfg.getint('eval', 'k')

    klist = [k for k in [1, 3, 5, 10, 25, 50, 100, 250, 500, 1000] if k <= eval_k]

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
        gtlist.append(get_rank_relevance(qlabel, rank, eval_k))
        print(gtlist[-1].shape, "\n---")

    gtarr = np.vstack(gtlist)
    print(gtarr.shape)
    aux = []

    for k in klist:
        aux.append(np.mean(gtarr[:, 0:k], axis=1).reshape((-1,1)))

    prectable = np.hstack(aux)
    prectable = np.hstack([np.arange(prectable.shape[0]).reshape(-1, 1), prectable])

    # Computing and stacking the average between all queries
    prectable = np.vstack([prectable, np.mean(prectable, axis=0)])

    outrelfname = outdir + retcfg['DEFAULT']['expname'] + lbl_suffix + ".irp_lbls.npy"
    outevalfname = outdir + retcfg['DEFAULT']['expname'] + "_eval.csv"

    hdr = "Q#," + ",".join(["P@{0:03d}".format(k) for k in klist])

    fmt = "%04d," + ",".join(["%0.3f" for _ in klist])

    np.save(outrelfname, gtarr[:, 0:kp])
    np.savetxt(outevalfname, prectable, header=hdr,
               fmt=fmt, delimiter=",")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("cfgfile", help="Retrieval configuration file", type=str)
    parser.add_argument("-s", "--suffix", help="Suffix for label files", type=str, default="")
    args = parser.parse_args()

    retcfg = cfgloader(args.cfgfile)

    evaluate_and_label(retcfg, args.suffix)
