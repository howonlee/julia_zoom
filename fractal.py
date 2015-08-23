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

def julia_quadratic(fn=lambda x: x * x, c=complex(0, 0.65)):
    re_min, re_max = -2.0, 2.0
    im_min, im_max = -2.0, 2.0
    w, h = 512, 512
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
                new_arr[im_idx, re_idx] = 50
            else:
                new_arr[im_idx, re_idx] = 0
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

def zlib_energy(arr):
    return len(zlib.compress(arr.tobytes(), 1))

def energy(arr):
    sx = sci_im.filters.prewitt(arr, axis=0)
    sy = sci_im.filters.sobel(arr, axis=1)
    processed = np.hypot(sx, sy)
    return np.sum(processed)

def diff_energy(arr):
    return np.sum(np.diff(arr, axis=0))

def generate_neighbors(best_arr):
    neighbors = []
    for x in xrange(3):
        r1, r2 = random.randint(0,best_arr.shape[0]-1), random.randint(0, best_arr.shape[0]-1)
        x_swap = best_arr.copy()
        x_swap[[r1, r2],:] = x_swap[[r2, r1],:]
        neighbors.append(x_swap)
    # taken out because we are only shuffling one dim currently

    #for y in xrange(3):
    #    r1, r2 = random.randint(0,best_arr.shape[1]-1), random.randint(0, best_arr.shape[1]-1)
    #    y_swap = best_arr.copy()
    #    y_swap[:,[r1, r2]] = y_swap[:,[r2, r1]]
    #    neighbors.append(y_swap)
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

def test_unscrambling():
    frac_arr = julia_quadratic()
    npr.shuffle(frac_arr)
    #npr.shuffle(frac_arr.T)
    unscramble(frac_arr)

def get_indiv_box_count(i, j, box_size, mat):
    #this is embarrasingly parallel
    for k in xrange(0, box_size):
        for l in xrange(0, box_size):
            if mat.has_key((i + k, j + l)):
                return 1
    return 0

def get_box_count(box_size, mat):
    count = 0
    for i in xrange(0, mat.shape[0], box_size):
        for j in xrange(0, mat.shape[0], box_size):
            count += get_indiv_box_count(i, j, box_size, mat)
    return count

def plot_box_counts(mat, name):
    """
    we can fit the lognormal distribution normally, I think
    """
    mat = sci_sp.dok_matrix(mat)
    box_sizes = list(reversed(range(3, 100)))
    box_counts = []
    for box_size in box_sizes:
        print "getting box size ", box_size
        box_counts.append(get_box_count(box_size, mat))
    plt.title("box size vs. box counts")
    plt.xlabel("box size")
    plt.ylabel("box count")
    plt.loglog(box_sizes, box_counts)
    plt.savefig(name)

def plot_fft(mat, name):
    fft_mat = np.fft.fft2(mat)
    plt.close()
    plt.imshow(np.log(np.abs(np.fft.fftshift(fft_mat))**2))
    plt.savefig(name + "_mag")

def plot_edges(mat, name):
    plt.close()
    sx = sci_im.filters.prewitt(mat, axis=0)
    sy = sci_im.filters.prewitt(mat, axis=1)
    processed = np.hypot(sx, sy)
    plt.imshow(processed)
    plt.savefig(name + "_edge")

def print_edge(mat):
    sx = sci_im.filters.sobel(mat, axis=0)
    sy = sci_im.filters.sobel(mat, axis=1)
    processed = np.hypot(sx, sy)
    print np.sum(processed)

def print_diff(mat):
    print np.sum(np.diff(mat))

def print_degree_stats(arr):
    npr.shuffle(frac_arr)
    npr.shuffle(frac_arr.T)
    degrees = arr.sum(axis=0)
    degrees = np.sort(degrees)
    plt.plot(degrees)
    plt.savefig("fract_stats")

if __name__ == "__main__":
    # fn_explore(lambda x: np.tanh(x), np.linspace(1, 3, 10))
    # explore(np.linspace(1, 3, 10))
    # scrambling is inplace
    # plot_fft_edges(frac_arr, "fft_unshuffled")
    # print_edge(frac_arr)
    frac_arr = julia_quadratic()
    print_degree_stats(frac_arr)
    #plt.imshow(frac_arr)
    #plt.show()
    #test_unscrambling()
