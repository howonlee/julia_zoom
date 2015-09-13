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
    """
    This will dominate the time
    Easiest first optimization: use the sparsity, luke
    """
    e = 0
    center = (arr.shape[0] // 2, arr.shape[1] // 2)
    for x in xrange(arr.shape[0]):
        for y in xrange(arr.shape[1]):
            if arr[x,y] == 1:
                e += dist to center ###########
    return e

def perform_swap(arr, swap):
    # performs swap in place
    return arr

def generate_neighbors(best_arr):
    swaps = generate some swaps ############
    arrs = []
    for swap in swaps:
        new_arr = best_arr.copy()
        arrs.append(perform_swap(new_arr, swap))
    return arrs

def scramble(arr):
    swaps = [(num1, num2) for num1, num2 in xrange(some shit)] ################
    new_arr = arr.copy()
    for swap in swaps:
        new_arr = perform_swap(new_arr, swap)
    return new_arr

def unscramble(scrambled_arr):
    pass #############333

def test_unscrambling():
    pass ################

if __name__ == "__main__":
    frac = julia_quadratic()
    plt.imshow(frac)
    plt.show()
