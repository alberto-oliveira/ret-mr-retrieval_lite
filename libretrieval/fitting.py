#!/usr/bin/env python
#-*- coding: utf-8 -*-

import argparse

import numpy as np

from libretrieval.utility import cfgloader
from libretrieval.engine import RetrievalEngine

import matlab.engine
matlab_engine = matlab.engine.start_matlab()


def fit_distribution(data, dbntype):
    data_m = matlab.double(data[data != 0].reshape(-1).tolist())

    distb = dict(name=dbntype)

    if dbntype == 'wbl':
        estpar, _ = matlab_engine.wblfit(data_m, nargout=2)
        distb['scale'] = estpar[0][0]
        distb['shape'] = estpar[0][1]
        distb['loc'] = 0

    elif dbntype == 'gev':
        estpar, _ = matlab_engine.gevfit(data_m, nargout=2)
        distb['scale'] = estpar[0][1]
        distb['shape'] = estpar[0][0]
        distb['loc'] = estpar[0][2]

    return distb


def fitting(retcfg):

    print("-- Starting ret. engine")
    reengine = RetrievalEngine(retcfg)
    print('--Done\n')

    t = retcfg.getint('fit', 't')

    outpath_prefx = "{outdir:s}{expname:s}_t{tval:d}".format(outdir=retcfg['path']['outdir'],
                                                             expname=retcfg['DEFAULT']['expname'],
                                                             tval=t)

    fitparams_w = np.zeros((reengine.ni, 3), dtype=np.float64)
    fitparams_g = np.zeros((reengine.ni, 3), dtype=np.float64)

    for i in range(reengine.ni):
        rank = reengine.search(i, True)
        d = fit_distribution(rank[t:], 'wbl')

        fitparams_w[i, 0] = d['shape']
        fitparams_w[i, 1] = d['scale']
        fitparams_w[i, 2] = d['loc']

        print('  --- WBL fit: shape:{shape:0.3f} | scale:{scale:0.3f} | loc:{loc:0.3f}'.format(shape=d['shape'],
                                                                                               scale=d['scale'],
                                                                                               loc=d['loc']))

        d = fit_distribution(rank[t:], 'gev')

        fitparams_g[i, 0] = d['shape']
        fitparams_g[i, 1] = d['scale']
        fitparams_g[i, 2] = d['loc']

        print('  --- GEV fit: shape:{shape:0.3f} | scale:{scale:0.3f} | loc:{loc:0.3f}'.format(shape=d['shape'],
                                                                                               scale=d['scale'],
                                                                                               loc=d['loc']))
        print('---')

    np.save(outpath_prefx + ".wbl", fitparams_w)
    np.save(outpath_prefx + ".gev", fitparams_g)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("cfgfile", help="Retrieval configuration file", type=str)
    args = parser.parse_args()

    retcfg = cfgloader(args.cfgfile)

    fitting(retcfg)