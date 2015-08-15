import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    #explore(np.linspace(1, 3, 100))
    fn_explore(lambda x: np.tanh(x), np.linspace(1, 3, 10))
