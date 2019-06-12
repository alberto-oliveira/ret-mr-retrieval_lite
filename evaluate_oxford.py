#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os

import numpy as np



outfile = sys.argv[1]
lblfile = sys.argv[2]
maxk = int(sys.argv[3])

klist = [k for k in [1, 3, 5, 10, 25, 50, 100, 250, 500] if k <= maxk]
n = len(klist) + 1

lbl = np.load(lblfile)

prectable = np.zeros((lbl.shape[0], n), dtype=np.float32)
prectable[:, 0] = np.arange(lbl.shape[0])


for i, row in enumerate(lbl):
    for j, k in enumerate(klist):

        # first column is numbering, thus why j+1
        prectable[i, j+1] = np.mean(row[:k])

prectable = np.vstack([prectable, np.mean(prectable, axis=0)])

hdr = "Q#," + ",".join(["P@{0:03d}".format(k) for k in klist])

fmt = "%04d," + ",".join(["%0.3f" for _ in klist])

np.savetxt(outfile, prectable, header=hdr,
           fmt=fmt, delimiter=",")