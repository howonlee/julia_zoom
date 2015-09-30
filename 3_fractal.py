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

def perform_swap(arr, swap):
    first, second = swap
    arr[:, [first, second]] = arr[:, [second, first]]
    arr[[first, second], :] = arr[[second, first], :]
    return arr

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

def get_prev(idx, arr):
    prev = idx - 1
    if prev < 0:
        prev = arr.shape[0] - 1
    return prev

def get_next(idx, arr):
    next_idx = idx + 1
    if next_idx >= arr.shape[0]:
        next_idx = 0
    return next_idx

def is_satisfied(idx, arr, thresh=0.90):
    prev_idx, next_idx = get_prev(idx, arr), get_next(idx, arr)
    concordances = 0
    for x in xrange(arr.shape[0]):
        if arr[next_idx, x] == arr[idx, x]:
            concordances += 1
        if arr[prev_idx, x] == arr[idx, x]:
            concordances += 1
    # there's a few off-by-ones but just have a big lattice, OK?
    ratio = float(concordances) / (arr.shape[0] * 2)
    return ratio > thresh

def random_nonsatisfied(satisfieds, chance=0.005):
    idx = 0
    while True:
        idx = random.randint(0, len(satisfieds) -1)
        if not satisfieds[idx]:
            break
        if random.random() < chance:
            break
    return idx

def unscramble(scrambled_arr, name="unscrambled_schelling", num_iters=200):
    """
    Schelling unscramble: automata is actually on 1d lattice cuz of that "sorting" effect
    """
    curr_arr = scrambled_arr.copy()
    satisfied = [False] * curr_arr.shape[0]
    print "starting..."
    best_score = -1
    best_arr = scrambled_arr.copy()
    for x in xrange(num_iters):
        score = sum(map(lambda x: 1 if x else 0, satisfied))
        print "current: ", score, " best: ", best_score
        if score > best_score:
            print "found best"
            best_score = score
            best_arr = curr_arr.copy()
        satisfied = [is_satisfied(idx, curr_arr) for idx in xrange(curr_arr.shape[0])]
        for x in xrange(curr_arr.shape[0]):
            prev_x, next_x = get_prev(x, curr_arr), get_next(x, curr_arr)
            if not satisfied[x] and (satisfied[prev_x] or satisfied[next_x]):
                swap = (x, random_nonsatisfied(satisfied))
                curr_arr = perform_swap(curr_arr, swap)
    plt.imshow(best_arr)
    plt.savefig("./pics/" + name + "_" + str(best_score))
    sys.exit(0)

def test_unscrambling(name):
    frac = julia_quadratic()
    scrambled_frac = scramble(frac)
    unscramble(scrambled_frac, name=name)

def test_satisfaction():
    frac = julia_quadratic()
    satisfied = [is_satisfied(idx, frac) for idx in xrange(frac.shape[0])]
    score = sum(map(lambda x: 1 if x else 0, satisfied))
    print score

if __name__ == "__main__":
    assert len(sys.argv) == 2
    test_unscrambling(sys.argv[1])
