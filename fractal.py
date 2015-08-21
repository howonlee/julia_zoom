import numpy as np
import numpy.random as npr
import numpy.linalg as np_lin
import matplotlib.pyplot as plt
import random
import numba
import operator
import lz4
import zlib
import sys

def julia_quadratic(fn=lambda x: x * x, c=complex(0, 0.65)):
    re_min, re_max = -2.0, 2.0
    im_min, im_max = -2.0, 2.0
    w, h = 512, 512
    real_range = np.arange(re_min, re_max, (re_max - re_min)/ w)
    imag_range = np.arange(im_min, im_max, (im_max - im_min)/ h)
    new_arr = np.zeros((len(imag_range), len(real_range)))
    for im_idx, im in enumerate(imag_range):
        for re_idx, re in enumerate(real_range):
###############
###############
###############
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

def lz4_energy(arr):
    return len(lz4.dumps(arr.tobytes()))

def energy(arr):
    return len(zlib.compress(arr.tobytes(), 1))

def generate_neighbors(best_arr):
    neighbors = []
    for x in xrange(3):
        r1, r2 = random.randint(0,best_arr.shape[0]-1), random.randint(0, best_arr.shape[0]-1)
        x_swap = best_arr.copy()
        x_swap[[r1, r2],:] = x_swap[[r2, r1],:]
        neighbors.append(x_swap)
    for y in xrange(3):
        r1, r2 = random.randint(0,best_arr.shape[1]-1), random.randint(0, best_arr.shape[1]-1)
        y_swap = best_arr.copy()
        y_swap[:,[r1, r2]] = y_swap[:,[r2, r1]]
        neighbors.append(y_swap)
    return neighbors


def unscramble(scrambled_arr):
    # currently just gradient descent
    best_arr = scrambled_arr.copy()
    best_energy = energy(best_arr)
    i = 0
    temperature = 1
    try:
        while True:
            i += 1
            if i % 20 == 0:
                print i
            neighbors = generate_neighbors(best_arr)
            for neighbor in neighbors:
                neighbor_energy = energy(neighbor)
                if neighbor_energy < best_energy:
                    print "found best"
                    best_energy = neighbor_energy
                    best_arr = neighbor
    except KeyboardInterrupt:
        plt.imshow(best_arr)
        plt.savefig("unscrambled.png")
        sys.exit(1)

if __name__ == "__main__":
    # fn_explore(lambda x: np.tanh(x), np.linspace(1, 3, 10))
    # explore(np.linspace(1, 3, 10))
    frac_arr = julia_quadratic()
    # scrambling is inplace
    npr.shuffle(frac_arr)
    npr.shuffle(frac_arr.T)
    unscramble(frac_arr)
