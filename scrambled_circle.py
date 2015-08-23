import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

def create_circle():
    xx, yy = np.mgrid[:1000, :1000]
    circle = (xx - 500) ** 2 + (yy - 500) ** 2
    circle = circle < 20000
    return circle

if __name__ == "__main__":
    circ = create_circle()
    # these are inplace
    npr.shuffle(circ)
    npr.shuffle(circ.T)
    plt.imshow(circ)
    plt.savefig("circ")
