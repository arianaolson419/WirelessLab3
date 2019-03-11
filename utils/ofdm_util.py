"""This file contains functions to implement OFDM for Lab 3.

Author: Ariana Olson
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

def create_signal_frequency_domain(num_samples, num_packets):
    """Create a bit sequence in the frequency domain.

    Args:
        num_samples (int): the number of samples in each packet of data.
        num_packets (int): the number of packets of data to transmit.
    
    Returns:
        signal_freq (1D float32 ndarray): the bit sequence in the frequency
            domain.
    """
    pass

def create_signal_time_domain(num_samples_data, num_samples_prefix, signal_freq):
    """Convert a sequence of frequency domain symbols into a time domain signal
    with cyclic prefixes.

    Args:
        num_samples_data (int): The number of samples of data in each packet.
        num_samples_prefix (int): The number of samples to use as the cyclic
            prefix of each packet.
        signal_freq (1D float32 ndarray): the bit sequence in the frequency
            domain.

    Returns:
        signal_time: A time domain signal of data packets with cyclic prefixes.
    """
    pass
