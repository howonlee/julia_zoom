import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

def create_circle():
    xx, yy = np.mgrid[:1000, :1000]
    circle = (xx - 500) ** 2 + (yy - 500) ** 2
    circle = circle < 20000
    return circle

def plot_shuffled(circ):
    npr.shuffle(circ)
    npr.shuffle(circ.T)
    plt.imshow(circ)
    plt.savefig("circ")

def circle_stats(circ):
    npr.shuffle(circ)
    npr.shuffle(circ.T)
    sums = circ.sum(axis=0)
    sums = np.sort(sums)
    plt.plot(sums)
    plt.savefig("circle_stats")

if __name__ == "__main__":
    circ = create_circle()
    circle_stats(circ)
    # these are inplace
