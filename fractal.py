import numpy as np
import matplotlib.pyplot as plt

def julia_quadratic(c=complex(0, 0.65)):
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
                z = z * z + c
                n -= 5
            new_arr[im_idx, re_idx] = n
    return new_arr

if __name__ == "__main__":
    arr = julia_quadratic()
    plt.matshow(arr)
    plt.show()
