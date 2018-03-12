# -*- coding: utf-8 -*-

import numpy as np
import cv2

def noise(img, noise_type="gaussian", **kwargs):

    mean = kwargs.get("mean", 0)
    var = kwargs.get("variance", 0.1)
    sp_ammount = kwargs.get("sp_ammount", 0.005)
    sp_prop = kwargs.get("sp_proportion", 0.5)

    noisy = np.copy(img)
    if noise_type == "gaussian":
        row, col, ch = noisy.shape
        sigma = var ** 0.5
        print("(mean={0:0.2f}; var={1:0.2f}) ".format(mean, var), end="")
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = noisy + gauss

    elif noise_type == "sp":
        row, col, ch = noisy.shape

        # Salt mode
        s_num = int(np.ceil(sp_ammount * noisy.size * sp_prop))
        p_num = int(np.ceil(sp_ammount * noisy.size * (1. - sp_prop)))

        pos = np.zeros(row*col)
        pos[0:s_num] = 1
        pos[s_num:s_num+p_num] = 2
        np.random.shuffle(pos)

        noisy[(pos == 1).reshape(row, col), :] = 0
        noisy[(pos == 2).reshape(row, col), :] = 255

    return noisy
