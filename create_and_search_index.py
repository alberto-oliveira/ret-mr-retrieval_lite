#!/usr/bin/env python
#-*- coding: utf-8 -*-

import argparse
import glob

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

from libretrieval.utility import safe_create_dir, cfgloader
from libretrieval.features.io import load_features

import time

import ipdb as pdb


def create_and_search_index(retcfg, jobs):

    q_features = load_features(retcfg['path']['qfeature'])
    q_namelist = np.loadtxt(retcfg['path']['qlist'], dtype=dict(names=('qname', 'nfeat'), formats=('U100', np.int32)))

    assert q_features.shape[0] == np.sum(q_namelist['nfeat']), "Inconsistent number of features sum and size of" \
                                                               "query features array"

    norm = retcfg.get('feature', 'norm', fallback=None)

    db_features = load_features(retcfg['path']['dbfeature'])
    if norm:
        db_features = normalize(db_features, norm)
        q_features = normalize(q_features, norm)

    outdir = retcfg['path']['outdir'] + "queryfiles/"
    safe_create_dir(outdir)

    index_type = retcfg['index']['index_type']
    dist_type = retcfg['index']['dist_type']

    knn = retcfg.getint('search', 'knn')

    print(" -- Creating <{0:s}> NN index".format(index_type))
    print("     -> KNN: {0:d}".format(knn))
    print("     -> Metric: {0:s}".format(dist_type))
    nnidx = NearestNeighbors(n_neighbors=knn, algorithm=index_type, metric=dist_type, n_jobs=jobs)
    nnidx.fit(db_features)

    ts = time.perf_counter()
    print("\n -- Searching index with {0:02d} jobs".format(jobs), flush=True)
    distances, indices = nnidx.kneighbors(q_features)
    print("    .Done (Elapsed = {0:0.3f}s".format(time.perf_counter() - ts), flush=True)

    s = 0
    for qname, n in q_namelist:

        qdists = distances[s:s+n]
        qidx = indices[s:s+n]

        matchfpath = "{0:s}{1:s}.matches".format(outdir, qname)
        distfpath = "{0:s}{1:s}.dist".format(outdir, qname)

        print(qname, "-> ", s, ":", s+n)
        print("   |_ dists: ", qdists.shape)
        print("   |_ indices: ", qidx.shape, end="\n---\n")
        np.save(matchfpath + ".npy", qidx)
        np.save(distfpath + ".npy", qdists)
        s += n

    print("---", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("cfgfile", help="Retrieval configuration file", type=str)
    parser.add_argument("-j", "--njobs", help="Number of concurrent jobs for searching", type=int, default=1)
    args = parser.parse_args()

    retcfg = cfgloader(args.cfgfile)

    create_and_search_index(retcfg, args.njobs)


