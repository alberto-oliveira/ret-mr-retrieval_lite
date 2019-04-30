#!/usr/bin/env python
#-*- coding: utf-8 -*-


import argparse
import glob

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

from libretrieval.utility import safe_create_dir, cfgloader
from libretrieval.features.io import load_features

from tqdm import tqdm

import ipdb as pdb

batch_size = 500


def create_and_search_db(retcfg, jobs):

    norm = retcfg.get('feature', 'norm', fallback=None)

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

    knn = retcfg.getint('search', 'knn')

    print(" -- Creating <{0:s}> NN index".format(index_type))
    print("     -> KNN: {0:d}".format(knn))
    print("     -> Metric: {0:s}\n".format(dist_type))

    mp = dict()
    if dist_type == 'seuclidean':
        mp['V'] = np.var(db_features, axis=0, dtype=np.float64)

    nnidx = NearestNeighbors(n_neighbors=knn, algorithm=index_type, metric=dist_type, n_jobs=jobs, metric_params=mp)
    nnidx.fit(db_features)

    indices = np.zeros((q_features.shape[0], knn), dtype=np.int32) - 1
    n_batches = int(np.ceil(q_features.shape[0] / batch_size))

    for i in tqdm(range(n_batches), ncols=100, desc='Batch #', total=n_batches):
        s = i*batch_size
        e = s + batch_size
        batch_q_features = q_features[s:e]
        indices_ = nnidx.kneighbors(batch_q_features, return_distance=False)

        assert indices[s:e].shape == indices_.shape, "output indices array shape <{0:s}> incompatible with batch indice" \
                                                     "array shape <{1:s}>".format(str(indices[s:e].shape), str(indices_))

        indices[s:e] = indices_

    assert np.argwhere(indices == -1).size == 0, "Indices on positions {0:s} have not been updated " \
                                                 "correctly".format(np.argwhere(indices == -1).tostring())

    outfile = "{0:s}{1:s}_db_matches.npy".format(outdir, retcfg.get('DEFAULT', 'expname'))
    np.save(outfile, indices)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("cfgfile", help="Retrieval configuration file", type=str)
    parser.add_argument("-j", "--njobs", help="Number of concurrent jobs for searching", type=int, default=1)
    args = parser.parse_args()

    retcfg = cfgloader(args.cfgfile)

    create_and_search_db(retcfg, args.njobs)
