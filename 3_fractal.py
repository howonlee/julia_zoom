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
import signal
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

def scramble(arr, num_swaps=5000):
    """
    Smarter way to do it wouth be the knuth shuffle
    who cares, tho
    """
    swaps = [(random.randint(0, arr.shape[0]-1), random.randint(0, arr.shape[0]-1)) for x in xrange(num_swaps)]
    new_arr = arr.copy()
    for swap in swaps:
        new_arr = perform_swap(new_arr, swap)
    return new_arr

def is_satisfied(idx, arr):
############### check if is satisfied with life
############### gonna have to have multiple iterations of this, I think
    pass

def unscramble(scrambled_arr, name="unscrambled_schelling"):
    """
    Schelling unscramble: automata is actually on 1d lattice cuz of that "sorting" effect
    """
    curr_arr = scrambled_arr.copy()
    satisfied = [is_satisfied(idx, curr_arr) for idx in xrange(curr_arr.shape[0])]
    while not all(satisfied):
        for x in xrange(curr_arr.shape[0]):
            if not satisfied[x]:
                swap with a random one #########
                maybe try to swap with dissatisfied more? ##########
    plt.imshow(curr_arr)
    plt.savefig("./pics/" + name)

def test_unscrambling():
    frac = julia_quadratic()
    scrambled_frac = scramble(frac)
    unscramble(scrambled_frac)

if __name__ == "__main__":
    assert len(sys.argv) == 2
    name = sys.argv[1]
    test_ga_unscrambling(name)
