#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import glob
import configparser

import cv2
import numpy as np
from sklearn.preprocessing import normalize

from libretrieval.utility import cfgloader, safe_create_dir
from libretrieval.search.query import flann_search
from libretrieval.ranking.score import *


def get_index_path(indexdir, prefix):

    aux = glob.glob(indexdir + "{0:s}*.dat".format(prefix))

    if len(aux) == 0:
        raise OSError("Index file with prefix <{0:s}> not found on directory <{1:s}>.".format(prefix, indexdir))
    else:
        return aux[0]


class RetrievalEngine:

    def __init__(self, cfg):

        if isinstance(cfg, str):
            self.retcfg = cfgloader(cfg)
        elif isinstance(cfg, configparser.ConfigParser):
            self.retcfg = cfg
        else:
            raise TypeError("Input parameter cfg must be either a path to a config file or a loaded configuration")

        self.dbnames = np.loadtxt(self.retcfg['path']['dblist'],
                                  dtype=dict(names=('name', 'nfeat'),
                                             formats=('U100', np.int32)))

        self.ni = self.dbnames['name'].shape[0]
        self.invidx = self.invert_index(self.dbnames)

        # Configuration parameters loaded into properties #
        # --- Feature Config --- #
        self.norm = self.retcfg.get('feature', 'norm', fallback=None)             # Type of feature normalization

        # --- Ranking Config --- #
        self.score = self.retcfg['rank']['score_type']                            # Score type {vote, distance}
        self.limit = self.retcfg.getint('rank', 'limit', fallback=np.newaxis)     # Limit # in rank

        # --- Search Config  --- #
        self.search_type = self.retcfg['search']['search_type']                   # Type of search {knn, radius}
        self.knn = self.retcfg.getint('search', 'knn')                            # Number of neighbors
        self.rfactor = self.retcfg.getfloat('search', 'radius_factor')            # Radius factor

        self.dbfeatures = self.load_features(self.retcfg['path']['dbfeature'], self.norm)

        self.flannIndex = cv2.flann_Index()
        ipath = get_index_path(self.retcfg['path']['outdir'], self.retcfg['DEFAULT']['expname'])
        self.flannIndex.load(self.dbfeatures, ipath)


    @staticmethod
    def load_features(p, norm):

        if os.path.isfile(p):
            features = np.load(p)
        elif os.path.isdir(p):

            filelist = glob.glob(p + "/*.npy")
            assert len(filelist) > 0, "No .npy files found on input directory"

            filelist.sort()

            featlist = [np.load(f) for f in filelist]
            features = np.vstack(featlist)

        else:
            raise OSError("<{0:s}> is neither file nor directory".format(p))

        if norm:
            features = normalize(features, norm)

        return features

    @staticmethod
    def invert_index(namelist):
        invidx = np.zeros(sum(namelist['nfeat']), dtype=np.uint32)

        sidx = 0
        for i, n in enumerate(namelist['nfeat']):
            eidx = sidx + n
            invidx[sidx:eidx] = i
            sidx = eidx

        return invidx

    def reload_index(self, ipath):

        self.flannIndex = cv2.flann_Index()
        self.flannIndex.load(self.dbfeatures, ipath)

        return

    def batch_search(self, batchfile, outdir, l=np.newaxis, numbered=False, verbose=True):

        if outdir[-1] != '/':
            outdir += '/'
        safe_create_dir(outdir)

        try:
            batch = np.loadtxt(batchfile, dtype=np.int32, usecols=0)
        except ValueError:
            batch = np.loadtxt(batchfile, dtype='U100', usecols=0)

        for i, query in enumerate(batch[:l]):
            print("batched query: ", query)

            rk = self.search(query, verbose)

            if numbered:
                self.write_rank(rk, query, outdir, prefix='{0:04d}'.format(i))
            else:
                self.write_rank(rk, query, outdir)

        return

    def search(self, query, verbose=False):

        if isinstance(query, np.int32) or isinstance(query, int):

            qi = query

            try:
                qname = self.dbnames['name'][qi]
            except IndexError:
                print(". Index <{0:d}> outside of dataset range {{0, {1:d}}}.\n".format(qi, self.ni))
                return

        elif isinstance(query, str):

            qname = query

            try:
                qi = np.flatnonzero(self.dbnames['name'] == qname)[0]  # Position
            except IndexError:
                print(". Query name <{0:s}> not found on dataset.\n".format(qname))
                return

        else:
            raise TypeError("\'query\' must be either a query name or a query index")

        fi = np.flatnonzero(self.invidx == qi)[0]
        nf = self.dbnames['nfeat'][qi]

        qfeatures = self.dbfeatures[fi:fi+nf]

        if verbose:
            print(". Searching: ", qname)
            print("   |_ Position: ", qi)
            print("   |_ # Features: ", nf, end="\n\n")

        votes, dists, _ = flann_search(qfeatures,
                                       self.flannIndex,
                                       stype=self.search_type,
                                       k=self.knn,
                                       f=self.rfactor,
                                       flib="cv")

        matchscores = np.zeros(self.ni, dtype=np.float64)
        distscores = np.zeros(self.ni, dtype=np.float64)
        namearray = self.dbnames['name']

        for i in range(nf):

            # v has the index of the feature that got the vote. invidx[v] is the index of the
            # collection image the feature v is from. Thus, 1 is summed in the number of votes
            # for the feature, and the distance got is summed as well (to be divided later)
            for v, d in zip(votes[i, :], dists[i, :]):
                dbi = self.invidx[v]
                matchscores[dbi] += 1
                distscores[dbi] += d

        if self.score == "vote":
            finalscore = matchscores
        elif self.score == "distance":
            finalscore = distscores/matchscores

        aux = np.logical_not(np.logical_or(np.isnan(finalscore), np.isinf(finalscore)))
        finalscore = finalscore[aux]
        namearray = namearray[aux]

        aux = zip(namearray, finalscore)

        dt = dict(names=('name', 'score'),
                  formats=('U100', np.float64))

        rank = np.array([a for a in aux], dtype=dt)
        rank.sort(order=('score', 'name'))

        if self.score == "vote":
            rank = rank[::-1]

        return rank[0:self.limit]

    @staticmethod
    def write_rank(rank, qname, outdir, prefix=''):

        if prefix != '':
            prefix += '_'

        rankfpath = "{outdir:s}{prefix:s}{qname:s}.rk".format(outdir=outdir, prefix=prefix, qname=qname)
        np.savetxt(rankfpath, rank, fmt="%-50s %10.5f")


