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

batch_size = 500


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
    print("     -> Metric: {0:s}\n".format(dist_type))

    mp = dict()
    if dist_type == 'seuclidean':
        mp['V'] = np.var(db_features, axis=0, dtype=np.float64)

    nnidx = NearestNeighbors(n_neighbors=knn, algorithm=index_type, metric=dist_type, n_jobs=jobs, metric_params=mp)
    nnidx.fit(db_features)

    distances = []
    indices = []
    n_batches = int(np.ceil(q_features.shape[0] / batch_size))

    for i in range(n_batches):
        s = i*batch_size
        e = s + batch_size
        batch_q_features = q_features[s:e]
        ts = time.perf_counter()
        print(" -- Searching batch {0:03d}/{1:03d} with {2:02d} jobs".format(i+1, n_batches, jobs), flush=True, end='')
        distances_, indices_ = nnidx.kneighbors(batch_q_features)
        print(" ...Done (Elapsed = {0:0.3f}s".format(time.perf_counter() - ts), flush=True)

        distances.append(distances_)
        indices.append(indices_)

    distances = np.vstack(distances)
    indices = np.vstack(indices)

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


