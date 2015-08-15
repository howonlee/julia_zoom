import numpy as np
import matplotlib.pyplot as plt

def generate_vals():
    re_min, re_max = -2.0, 2.0
    im_min, im_max = -2.0, 2.0
    real_range = np.arange(re_min, re_max, (re_max - re_min))
    imag_range = np.arange(im_min, im_max, (im_max - im_min))
    arr = something ############
    return arr

def julia_quadratic(arr, c=complex(0, 0.75)):
    ##############3
    new_arr = arr.copy()
    for im in image_range:
        for re in real_range:
            z = complex(re, im)
            n = 250
            while abs(z) < 10 and n > 0:
                z = z * z + c
                n -= 1
            new_arr[imthing, rething] = n
    return new_arr

if __name__ == "__main__":
    arr = generate_vals()
    arr = julia_quadratic(arr)
    plt.matshow(arr)
