import numpy as np
import numpy.random as npr
import numpy.linalg as np_lin
import matplotlib.pyplot as plt
import random
import numba

def julia_quadratic(fn=lambda x: x * x, c=complex(0, 0.65)):
    re_min, re_max = -2.0, 2.0
    im_min, im_max = -2.0, 2.0
    w, h = 512, 512
    real_range = np.arange(re_min, re_max, (re_max - re_min)/ w)
    imag_range = np.arange(im_min, im_max, (im_max - im_min)/ h)
    new_arr = np.zeros((len(imag_range), len(real_range)))
    for im_idx, im in enumerate(imag_range):
        for re_idx, re in enumerate(real_range):
            z = complex(re, im)
            n = 250
            while abs(z) < 10 and n > 50:
                z = fn(z) + c
                n -= 5
            new_arr[im_idx, re_idx] = n
    return new_arr

def explore(exponents):
    for exp in exponents:
        plt.close()
        new_fn = lambda x: x ** exp
        arr = julia_quadratic(fn=new_fn)
        plt.matshow(arr)
        plt.savefig(str(exp) + ".png")

def fn_explore(new_fn, cvals):
    for cval in cvals:
        plt.close()
        arr = julia_quadratic(fn=new_fn, c=cval)
        plt.matshow(arr)
        plt.savefig(str(cval) + ".png")

def energy(arr):
    # this is the inner loopiest inner loop I've
    # we need to vectorize the shit out of this
    # - or - translate to julia
    potential = 0
    print "energy calculation..."
    for orig_x in xrange(arr.shape[0]):
        for orig_y in xrange(arr.shape[1]):
            for compare_x in xrange(arr.shape[0]):
                for compare_y in xrange(arr.shape[1]):
                    if orig_x == compare_x or compare_x == compare_y:
                        continue
                    # potential += negatives and positives and shit / (l2 norm of that shit)
    # possible other shit to try:
    # point charges in blank space (as opposed to + and - charges competing)
    # both of the above with r2 distance
    # note the isotropy wrt symmetries here, maybe something to flip?
    return potential

def unscramble(scrambled_arr, num_iters=500):
    best_arr = scrambled_arr.copy()
    curr_arr = scrambled_arr.copy()
    best_energy = energy(best_arr)
    curr_energy = energy(curr_arr)
    i = 0
    while i < num_iters:
        # generate some pairs of coords to swap
        # swap on either x or y axis, see if energy good
        # don't do the SA temperature yet
        pass
    return best_arr, best_energy

if __name__ == "__main__":
    # fn_explore(lambda x: np.tanh(x), np.linspace(1, 3, 10))
    # explore(np.linspace(1, 3, 100))
    frac_arr = julia_quadratic()
    # scrambling is inplace
    npr.shuffle(frac_arr)
    npr.shuffle(frac_arr.T)
    plt.imshow(frac_arr)
    plt.show()
