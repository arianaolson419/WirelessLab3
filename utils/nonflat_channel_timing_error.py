import time

def nonflat_channel(x):
    import numpy as np

    h = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.1912, 0.9316, 0.2821, -0.1990, 0.1630, -0.1017, 0.0544, -0.0261, 0.0090, 0.0000, -0.0034, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ])
    # convolve the signal with the impulse response of the channel
    y = np.convolve(x, h)
    np.random.seed(int(time.time()))
    f_delta = np.pi*np.random.uniform(0.0, 1.0/256.0)
    print(f_delta, 'sim f_delta')
    y = y*np.exp(1j*f_delta*np.linspace(0, y.size-1,y.size))
    # get the dimensions of the resulting array
    shape_y = np.shape(y)
    # add noise of a specified variance, this value gives an SNR of about 30dB
    n = 0.0087*np.random.normal(0, 1, shape_y)+0.0087j*(np.random.normal(0, 1, shape_y))

    return y+n, f_delta
