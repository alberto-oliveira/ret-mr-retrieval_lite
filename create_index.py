#!/usr/bin/env python
#-*- coding: utf-8 -*-

import argparse

from libretrieval.utility import safe_create_dir, cfgloader
from libretrieval.features.io import load_features

from libretrieval.search.index import create_flann_index

def create_index_(retcfg):

    index_type = retcfg['index']['index_type']
    dist_type = retcfg['index']['dist_type']
    lib = retcfg['index']['lib']
    norm = retcfg.get('feature', 'norm', fallback=None)

    safe_create_dir(retcfg['path']['outdir'])

    dbfeatures = load_features(retcfg['path']['dbfeature'])

    outpath = "{0:s}{1:s}_{2:s}_{3:s}.dat".format(retcfg['path']['outdir'], retcfg['DEFAULT']['expname'],
                                                  index_type, dist_type)

    if lib == "cv":
        fidx, params = create_flann_index(dbfeatures, index_type, dist_type, norm=norm)
    else:
        raise ValueError("Unsupported flann lib <{0:s}>".format(lib))

    fidx.save(outpath)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("cfgfile", help="Retrieval configuration file", type=str)
    args = parser.parse_args()

    retcfg = cfgloader(args.cfgfile)

    create_index(retcfg)
