#!/usr/bin/env python
#-*- coding: utf-8 -*-


import argparse, os
import glob

import psutil

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

from libretrieval.utility import safe_create_dir, cfgloader
from libretrieval.features.io import load_features

from tqdm import tqdm

import nmslib

#import ipdb as pdb

batch_size = 500


def memuse(order=0):
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
    print('{0:d}: memory use: {1:0.3f}GB'.format(order, memoryUse))

def check_consistency(a, k):

    print("Checking consistency...")
    for i in tqdm(range(len(a)), ncols=100, desc="Res", total=len(a)):
        elm = a[i]
        if elm.size != k:
            raise ValueError("Element [{0:d}] size <{1:d}> incompatible with k = <{2:d}>".format(i, elm.size, k))


def create_and_search_db(retcfg, jobs):

    batch_size = -1

    norm = retcfg.get('feature', 'norm', fallback=None)


    print(" -- loading DB features from: {0:s}".format(retcfg['path']['dbfeature']))
    db_features = load_features(retcfg['path']['dbfeature'])
    # We are creating an array of topk for the DB objects
    q_features = db_features

    if norm:
        db_features = normalize(db_features, norm)
        q_features = normalize(q_features, norm)

    outdir = retcfg['path']['outdir']
    safe_create_dir(outdir)

    index_type = retcfg['index']['index_type']
    dist_type = retcfg['index']['dist_type']

    knn = retcfg.getint('search', 'knn_db', fallback=300)
    M = retcfg.getint('index', 'M', fallback=100)
    efC = retcfg.getint('index', 'efC', fallback=100)

    print(" -- Creating <{0:s}> NN index".format(index_type))
    print("     -> KNN: {0:d}".format(knn))
    print("     -> Metric: {0:s}\n".format(dist_type))

    nnidx = nmslib.init(method=index_type, space=dist_type)
    nnidx.addDataPointBatch(db_features)
    del db_features
    nnidx.createIndex({'post': 2, 'efConstruction': efC, 'M': M}, print_progress=True)
    nnidx.setQueryTimeParams({'efSearch': knn})

    if batch_size == -1:
        batch_size = q_features.shape[0]

    n_batches = int(np.ceil(q_features.shape[0] / batch_size))

    for i in tqdm(range(n_batches), ncols=100, desc='Batch #', total=n_batches):
        s = i*batch_size
        e = s + batch_size
        batch_q_features = q_features[s:e]

        neighbours = nnidx.knnQueryBatch(batch_q_features, k=knn, num_threads=jobs)
        neighbours = list(zip(*neighbours))

    #pdb.set_trace()

    check_consistency(neighbours[0], knn)
    check_consistency(neighbours[1], knn)

    outfile = "{0:s}{1:s}_db_matches.npy".format(outdir, retcfg.get('DEFAULT', 'expname'))
    np.save(outfile, np.array(neighbours[0]))

    outfile = "{0:s}{1:s}_db_scores.npy".format(outdir, retcfg.get('DEFAULT', 'expname'))
    np.save(outfile, np.array(neighbours[1]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("cfgfile", help="Retrieval configuration file", type=str)
    parser.add_argument("-j", "--njobs", help="Number of concurrent jobs for searching", type=int, default=1)
    args = parser.parse_args()

    print(" -- Loading cfg")
    retcfg = cfgloader(args.cfgfile)

    create_and_search_db(retcfg, args.njobs)
