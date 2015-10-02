import numpy as np
import numpy.random as npr
import numpy.linalg as np_lin
import matplotlib.pyplot as plt
import scipy.sparse as sci_sp
import scipy.ndimage as sci_im
import networkx as nx
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

def zlib_energy(arr):
    return len(zlib.compress(arr.tobytes(), 1))

def lz4_energy(arr):
    return len(lz4.dumps(arr.tobytes()))

def gzip_energy(arr):
    raise NotImplementedError()

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
                e += np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                # taxi distance, because it's easy
    return e

def perform_swap(arr, swap):
    first, second = swap
    arr[:, [first, second]] = arr[:, [second, first]]
    arr[[first, second], :] = arr[[second, first], :]
    return arr

def generate_neighbors(best_arr):
    swaps = [(random.randint(0, best_arr.shape[0]-1), random.randint(0, best_arr.shape[0]-1)) for x in xrange(10)]
    arrs = []
    for swap in swaps:
        new_arr = best_arr.copy()
        arrs.append(perform_swap(new_arr, swap))
    return arrs

def scramble(arr, num_swaps=5000):
    """
    Smarter way to do it wouth be the knuth shuffle
    """
    swaps = [(random.randint(0, arr.shape[0]-1), random.randint(0, arr.shape[0]-1)) for x in xrange(num_swaps)]
    new_arr = arr.copy()
    for swap in swaps:
        new_arr = perform_swap(new_arr, swap)
    return new_arr

def fuckit_energy(arr, orig_arr):
    return np.sum(np.abs(arr - orig_arr))

def unscramble(scrambled_arr, orig_arr, num_iters=20000):
    """
    Let's have some proper tau-EO
    """
    best_energy = float("inf")
    best_arr = scrambled_arr.copy()
    for i in xrange(num_iters):
        neighbors = generate_neighbors(best_arr)
        for neighbor in neighbors:
            neighbor_energy = fuckit_energy(neighbor, orig_arr)
            if neighbor_energy < best_energy:
                print "found best"
                print best_energy
                best_energy = neighbor_energy
                best_arr = neighbor
    plt.imshow(best_arr)
    plt.savefig("pics/unscrambled_2.png")
    sys.exit(0)

def ga_unscramble(scrambled_frac, num_agents=10, num_iters=100000, name="ga_unscramble"):
    agents = [scrambled_frac] * num_agents
    def make_swaps(agent):
        return (random.randint(0, agent.shape[0]-1),random.randint(0, agent.shape[0]-1))
    for i in xrange(num_iters):
        if i % 200 == 0:
            print i
        swaps = [make_swaps(agents[0]) for x in xrange(num_agents)]
        agents = [perform_swap(agent, swap) for agent, swap in zip(agents, swaps)]
        energies = np.array([lz4_energy(agent) for agent in agents])
        agents = [agents[energies.argmax()]] * 10
    for x in xrange(num_agents):
        plt.close()
        plt.imshow(agents[x])
        plt.savefig("pics/" + name + "_" + str(x))
    sys.exit(0)

def test_unscrambling():
    frac = julia_quadratic()
    scrambled_frac = scramble(frac)
    unscramble(scrambled_frac, frac)

def test_ga_unscrambling(name):
    frac = julia_quadratic()
    scrambled_frac = scramble(frac)
    ga_unscramble(scrambled_frac, name=name)

def test_scrambling_gzip_size():
    arr = julia_quadratic()
    print len(zlib.compress(arr.tobytes(), 1))
    samples = [0] * 10
    num_samples = 100
    for x in xrange(10):
        print x
        for sample in xrange(num_samples):
            scrambled_arr = scramble(arr, num_swaps=x * 10)
            samples[x] += len(zlib.compress(scrambled_arr.tobytes(), 1))
    print [float(member) / float(num_samples) for member in samples]

def test_add_julia_quadratic():
    first = np.zeros((512, 512))
    for x in np.arange(0.01, 1.00, 0.10):
        print x
        first += julia_quadratic(fn=lambda x: 1 / 1 + np.exp(-x), c=complex(0, x))
    first = julia_quadratic(fn=lambda x: x * (1 - x), c=1.2, size=512)
    plt.imshow(first)
    plt.show()

def read_data(filename="./corpus.txt"):
    with open(filename) as corpus_file:
        corpus = corpus_file.read().split()
        bigrams = zip(corpus, corpus[1:])
        net = nx.Graph()
        net.add_edges_from(bigrams)
        net_arr = nx.to_numpy_matrix(net)
        scramble(net_arr, num_swaps=20000)
    data_arr = net_arr[0:512, 0:512]
    return data_arr

def test_data_unscrambling():
    frac = julia_quadratic()
    data = read_data()
    unscramble(data, frac)

if __name__ == "__main__":
    test_data_unscrambling()
