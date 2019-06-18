#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import glob
import argparse

import numpy as np

import cv2

from libretrieval.utility import safe_create_dir

from tqdm import tqdm

completedir = lambda x: x + "/" if x[-1] != '/' else x

def get_detector(d, n=500):

    if d == "SURF":
        return cv2.xfeatures2d.SURF_create(hessianThreshold=30)
    if d == "SIFT":
            return cv2.xfeatures2d.SIFT_create(nfeatures=n)
    else:
        raise ValueError("Incorrect detector", d)


def get_descriptor(d, n=500):

    if d == "SURF":
        return cv2.xfeatures2d.SURF_create(hessianThreshold=30)
    if d == "SIFT":
            return cv2.xfeatures2d.SIFT_create(nfeatures=n)
    else:
        raise ValueError("Incorrect descriptor", d)

def gen_random_feat(sample):

    s = sample.shape
    dt = sample.dtype
    ma = sample.max()

    rfeat = np.random.randn(*s)*ma

    return rfeat.astype(dt)


def extract_features(inputdir, prefix, detector, descriptor, limit):

    safe_create_dir(os.path.dirname(prefix))

    impathlist = glob.glob(inputdir + "*")
    impathlist.sort()

    feat_per_img = []

    det = get_detector(detector, n=limit)
    des = get_descriptor(descriptor, n=limit)

    previous = None

    for i in tqdm(range(len(impathlist)), ncols=100, desc="Image:", total=len(impathlist)):

        impath = impathlist[i]
        basename = os.path.basename(impath)

        img = cv2.imread(impath)
        kp = det.detect(img, None)
        _, features = des.compute(img, kp)

        try:
            features = features[:limit]
        except TypeError:
            # For corrupted images generate some random features
            np.random.shuffle(previous)
            features = previous

        feat_per_img.append((basename, features.shape[0]))

        outfeatfile = "{0:s}_{1:s}_batch{2:06d}.npy".format(prefix, descriptor, i)
        np.save(outfeatfile, features)

        previous = features.copy()

    idx_dtype = dict(names=('name', 'nfeat'), formats=('U100', np.int32))

    outidxfile = "{0:s}_idx.out".format(prefix)
    np.savetxt(outidxfile, np.array(feat_per_img, dtype=idx_dtype), fmt="%-50s %d")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("inputdir", help="Input directory")
    parser.add_argument("prefix", help="Output directory")

    parser.add_argument("--detector", "-d",
                        help="Local Feature detector used.",
                        type=str,
                        choices=["SURF", "SIFT", "ORB", "DENSE", "BRISK"],
                        default="SURF")

    parser.add_argument("--descriptor", "-f",
                        help="Local descriptor used.",
                        type=str,
                        choices=["SURF", "SIFT", "ORB", "RootSIFT", "RootSURF", "BRISK"],
                        default="SURF")

    parser.add_argument("--limit", "-l",
                        help="Maximum number of keypoints to be detected by sparse detectors.",
                        type=int,
                        default=1000)

    args = parser.parse_args()

    inputdir = completedir(args.inputdir)
    prefix = args.prefix
    detector = args.detector
    descriptor = args.descriptor
    limit = int(args.limit)

    extract_features(inputdir, prefix, detector, descriptor, limit)