def nonflat_channel(x):    
    import numpy as np

# h is the impulse response of the channel, 64 samples are given, and most of the energy is contained within aprroximately 17 samples, enabling the use of a 16 sample cyclic prefix
    h = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.2623, -0.1000, 0.5759, 1.0000, 0.5549, -0.1000, -0.1757, 0.0500, 0.0788, -0.0100, -0.0240, -0.0000, 0.0041, 0.0000, -0.0007, -0.0000, 0.0001, 0.0000, -0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
# convolve the signal with the impulse response of the channel
    y = np.convolve(x, h)
# get the dimensions of the resulting array
    shape_y = np.shape(y) 
# add noise of a specified variance, this value gives an SNR of about 30dB
    n = 0.0087*np.random.normal(0, 1, shape_y)+0.0087j*(np.random.normal(0, 1, shape_y))
    
    return y
 
