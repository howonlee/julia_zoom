import numpy as np
import numpy.random as npr
import numpy.linalg as np_lin
import matplotlib.pyplot as plt
import scipy.sparse as sci_sp
import scipy.ndimage as sci_im
import random
import numba
import operator
import lz4
import zlib
import sys
import time

def julia_quadratic(fn=lambda x: x * x, c=complex(0, 0.65), size=512):
    re_min, re_max = -2.0, 2.0
    im_min, im_max = -2.0, 2.0
    w, h = size, size
    real_range = np.arange(re_min, re_max, (re_max - re_min)/ w)
    imag_range = np.arange(im_min, im_max, (im_max - im_min)/ h)
    new_arr = np.zeros((len(imag_range), len(real_range)))
    for im_idx, im in enumerate(imag_range):
        for re_idx, re in enumerate(real_range):
            # deal with the fact that we have equipotentials
            # although we might not want them
            z = complex(re, im)
            n = 250
            while abs(z) < 10 and n > 50:
                z = fn(z) + c
                n -= 5
            if n == 50:
                new_arr[im_idx, re_idx] = 1
            else:
                new_arr[im_idx, re_idx] = 0
    return new_arr

def energy(arr):
    e = 0
    center = some point ###############
    for every point in arr: ###############
        if point val == 1: ###########3##
            e += dist ##########3
    return e

def generate_neighbors(best_arr):
    pass ################

def unscramble(scrambled_arr):
    pass #############333

def test_unscrambling():
    pass ################

if __name__ == "__main__":
    frac = julia_quadratic()
    plt.imshow(frac)
    plt.show()
