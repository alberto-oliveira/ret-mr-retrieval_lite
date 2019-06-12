#!/usr/bin/env python
#-*- coding: utf-8 -*-


import os, gc
import argparse
import glob

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

from libretrieval.utility import safe_create_dir, cfgloader
from libretrieval.features.io import load_features

from tqdm import tqdm

import nmslib

import psutil

def memuse():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
    print('memory use:', memoryUse)


def create_and_search_index(retcfg, jobs):

    batch_size = -1

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

    nnidx = nmslib.init(method=index_type, space=dist_type)
    nnidx.addDataPointBatch(db_features)
    del db_features
    nnidx.createIndex({'post': 2, 'efConstruction': knn, 'M': knn}, print_progress=True)
    nnidx.setQueryTimeParams({'efSearch': knn})

    if batch_size == -1:
        batch_size = q_features.shape[0]

    n_batches = int(np.ceil(q_features.shape[0] / batch_size))

    c = 0
    for i in tqdm(range(n_batches), ncols=100, desc='Batch', total=n_batches):
        s = i*batch_size
        e = s + batch_size
        batch_q_features = q_features[s:e]

        neighbours = nnidx.knnQueryBatch(batch_q_features, k=knn, num_threads=jobs)

        for j in tqdm(range(len(neighbours)), ncols=100, desc='Saving', total=len(neighbours)):

            indices, dists = neighbours[j]
            qname, _ = q_namelist[c]

            if dists.size != knn:
                #raise ValueError("{0:d}:{1:d}:{2:s} -- all dists == 0".format(c, j, qname))
                pdb.set_trace()

            matchfpath = "{0:s}{1:s}.matches".format(outdir, qname)
            distfpath = "{0:s}{1:s}.dist".format(outdir, qname)

            np.save(matchfpath + ".npy", indices)
            np.save(distfpath + ".npy", dists)

            c += 1

    print("---", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("cfgfile", help="Retrieval configuration file", type=str)
    parser.add_argument("-j", "--njobs", help="Number of concurrent jobs for searching", type=int, default=1)
    args = parser.parse_args()

    retcfg = cfgloader(args.cfgfile)

    create_and_search_index(retcfg, args.njobs)


