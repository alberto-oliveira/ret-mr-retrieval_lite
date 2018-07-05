#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import glob

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

    def __init__(self, cfgfile):

        self.retcfg = cfgloader(cfgfile)
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
        self.ef = self.retcfg.getboolean('rank', 'exclude_first')                 # Exclude first
        self.ez = self.retcfg.getboolean('rank', 'exclude_zero')                  # Exclude zeroes

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

    def search_batch(self, batchfile, outdir, l=np.newaxis, numbered=False, verbose=True):

        try:
            batch = np.loadtxt(batchfile, dtype=np.int32, usecols=0)
        except ValueError:
            batch = np.loadtxt(batchfile, dtype='U100', usecols=0)

        for i, query in enumerate(batch[:l]):
            print("batched query: ", query)

            if numbered:
                self.search_image(query, outdir, '{0:04d}'.format(i), verbose)
            else:
                self.search_image(query, outdir, '', verbose)

        return

    def search_image(self, query, outdir, prefix='', verbose=False):

        safe_create_dir(outdir)

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

        votescores = np.zeros(self.ni, dtype=np.float32)
        distscores = np.zeros(self.ni, dtype=np.float32)
        namearray = self.dbnames['name']

        for i in range(nf):

            # v has the index of the feature that got the vote. invidx[v] is the index of the
            # collection image the feature v is from. Thus, 1 is summed in the number of votes
            # for the feature, and the distance got is summed as well (to be divided later)
            for v, d in zip(votes[i, :], dists[i, :]):
                dbi = self.invidx[v]
                votescores[dbi] += 1
                distscores[dbi] += d

        aux = np.flatnonzero(votescores)
        votescores = votescores[aux]
        distscores = distscores[aux]
        namearray = namearray[aux]

        distscores = distscores / votescores

        normvotes, _ = normalize_scores(votescores.reshape(-1, 1), minmax_range=(1, 2), cvt_sim=False)
        normdists, _ = normalize_scores(distscores.reshape(-1, 1), minmax_range=(1, 2), cvt_sim=True)

        normvotes = normvotes.reshape(-1)
        normdists = normdists.reshape(-1)

        # Excludes first result. Used to exclude the query image, in case it is present in the database.
        if self.ef:
            mv = np.max(votescores)
            aux = votescores != mv
            votescores = votescores[aux]
            distscores = distscores[aux]
            namearray = namearray[aux]

        # Excludes any results with 0 votes.
        if self.ez:
            aux = votescores != 0
            votescores = votescores[aux]
            distscores = distscores[aux]
            namearray = namearray[aux]

        aux = zip(namearray, votescores, normvotes, distscores, normdists)

        dt = dict(names=('name', 'votes', 'normv', 'dists', 'normd'),
                  formats=('U100', np.float32, np.float32, np.float32, np.float32))

        rank = np.array([a for a in aux], dtype=dt)

        # norm distances are used to sort, instead of pure mean distances, as they are normalized to a similarity,
        # thus being being in accord to voting, which is also a similarity.
        if self.score == "vote":
            rank.sort(order=('votes', 'normd', 'name'))
            rank = rank[::-1]  # Just inverting the sort order to highest scores

        elif self.score == "distance":
            rank.sort(order=('normd', 'votes', 'name'))
            rank = rank[::-1]  # Just inverting the sort order to highest scores

        if prefix != '':
            prefix += '_'

        rankfpath = "{0:s}{1:s}{2:s}.rk".format(outdir, prefix, qname)
        np.savetxt(rankfpath, rank[0:self.limit], fmt="%-50s %10.5f %10.5f %10.5f %10.5f")


