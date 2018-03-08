#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import argparse
import glob

import cv2

completedir = lambda d: d if d[-1] == "/" else d + "/"


def query_perturbations(inputdir, outdir, np):

    imgpathlist = glob.glob(inputdir + "*")
    imgpathlist.sort()

    for imgpath in imgpathlist:
        cv2.



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("inputdir", help="Directory with input images")
    parser.add_argument("outdir", help="Output directory for processed images")
    parser.add_argument("np", help="Number of pertrubations per image")

    args = parser.parse_args()

    query_perturbations(completedir(args.inputdir), completedir(args.outdir), np)