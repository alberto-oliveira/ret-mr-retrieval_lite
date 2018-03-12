#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import argparse
import glob

import cv2
import numpy as np

from libretrieval.postprocessing import noise
from libretrieval.utility import safe_create_dir

np.random.seed(93311)

completedir = lambda d: d if d[-1] == "/" else d + "/"

perturbations = ["sp", "gaussian", "blur", "rotate", "contrast", "brightness"]

def perturbate(img, ptype):

    if ptype == "sp":
        params = dict(sp_ammount=np.random.ranf()*0.01,
                      sp_proportion=((0.7 - 0.3)*np.random.ranf() + 0.3))

        img = noise(img, "sp", **params)

    if ptype == "gaussian":
        params = dict(mean=(10*np.random.ranf() - 5),
                      variance=(5.0*np.random.ranf()))

        img = noise(img, "gaussian", **params)

    if ptype == "blur":

        img = cv2.GaussianBlur(img, ksize=((2*np.random.randint(0, 3) + 1), (2*np.random.randint(0, 3) + 1)),
                                    sigmaX=2*np.random.ranf(),
                                    sigmaY=2*np.random.ranf(),
                                    dst=None)

    if ptype == "rotate":

        rows, cols, _ = img.shape

        M = cv2.getRotationMatrix2D(center=(cols/2, rows/2),
                                    angle=np.random.randint(-5, 5, 1)[0],
                                    scale=1)
        img = cv2.warpAffine(img, M, (cols, rows))

    if ptype == "brightness":

        b = np.random.randint(-50, 51, 1)
        img = np.clip(img + b, 0, 255).astype(np.uint8)

    if ptype == "contrast":

        img = np.clip(img*(1.0*np.random.ranf()+0.5), 0, 255).astype(np.uint8)

    return img




def query_perturbations(inputdir, outdir, nptb):

    imgpathlist = glob.glob(inputdir + "*")
    imgpathlist.sort()

    safe_create_dir(outdir)

    count = 0
    for imgpath in imgpathlist:
        img = cv2.imread(imgpath)
        parts = os.path.basename(imgpath).split("_", 1)
        basename, ext = parts[1].rsplit('.', 1)

        for k in range(nptb):
            outpath = "{0:s}{1:05d}_{2:s}_p{3:03d}.{4:s}".format(outdir, count, basename, k, ext)

            n = np.random.randint(1, 4, 1)[0]
            np.random.shuffle(perturbations)

            print(". {0:s} -> {1:s}".format(os.path.basename(imgpath), os.path.basename(outpath)))
            print("    |_ ", end="", flush=True)
            for pert in perturbations:
                print(" {0:s};".format(pert), end="", flush=True)
                img = perturbate(img, pert)

            print("", end="\n")
            cv2.imwrite(outpath, img)
            count += 1






if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("inputdir", help="Directory with input images")
    parser.add_argument("outdir", help="Output directory for processed images")
    parser.add_argument("np", help="Number of pertrubations per image", type=int)

    args = parser.parse_args()

    query_perturbations(completedir(args.inputdir), completedir(args.outdir), args.np)