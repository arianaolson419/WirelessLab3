"""This file contains functions to implement OFDM for Lab 3.

Author: Ariana Olson
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

# Define constants used in this implementation.
NUM_SAMPLES_PER_PACKET = 64
NUM_PACKETS = 10
NUM_SAMPLES_CYCLIC_PREFIX = 16

def create_signal_freq_domain(num_samples, num_packets, seed):
    """Create a bit sequence in the frequency domain.

    Args:
        num_samples (int): The number of samples in each packet of data.
        num_packets (int): The number of packets of data to transmit.
        seed (int): The seed for the random number generator.
    
    Returns:
        signal_freq (1D float32 ndarray): The bit sequence in the frequency
            domain. This will be of length num_packets * num_samples.
    """
    np.random.seed(seed)
    signal_freq = np.sign(np.random.randn(num_samples * num_packets))
    return signal_freq

def create_signal_time_domain(num_samples_data, num_samples_prefix, signal_freq):
    """Convert a sequence of frequency domain symbols into a time domain signal
    with cyclic prefixes.

    Args:
        num_samples_data (int): The number of samples of data in each packet.
        num_samples_prefix (int): The number of samples to use as the cyclic
            prefix of each packet.
        signal_freq (1D float32 ndarray): The bit sequence in the frequency
            domain.

    Returns:
        signal_time: A time domain signal of data packets with cyclic prefixes.
    """
    num_packets = signal_freq.shape[-1] // num_samples_data
    num_samples_per_packet = num_samples_data + num_samples_prefix
    num_time_samples = num_samples_per_packet  * num_packets
    signal_time = np.zeros(num_time_samples, dtype=np.complex64)

    for i in range(num_packets):
        packet_start = i * num_samples_data
        packet_end = packet_start + num_samples_data

        time_data_packet = np.fft.ifft(signal_freq[packet_start:packet_end])
        prefix = time_data_packet[-num_samples_prefix:]

        # Add prefix.
        prefix_start = i * num_samples_per_packet
        prefix_end = prefix_start + num_samples_prefix

        data_start = prefix_end
        data_end = prefix_end + num_samples_data
        
        signal_time[prefix_start:prefix_end] = prefix
        signal_time[data_start:data_end] = time_data_packet

    return signal_time

def estimate_channel(tx_known_signals_frequency, rx_known_signals_frequncy):
    """Estimate the frequency domain channel coefficient from a set of transmitted and received known signals.

    Args:
        tx_known_signals_frequency (list of 1D ndarrays): Frequency domain known signals that were transmitted through the channel.
        rx_known_signals_frequency (list of 1D ndarrays): The frequency domain received signals corresponding to the signals in tx_known_signals_frequency.

    Returns:
        H (complex float): the estimated channel in the frequency domain.
    """
    assert len(tx_known_signals_frequency) == len(rx_known_signals_frequncy)
    assert len(tx_known_signals_frequency) > 0

    Hs = []
    for tx, rx in zip(tx_known_signals_frequency, rx_known_signals_frequncy):
        Hs.append(np.mean(rx / tx))

    H = np.mean(np.array(Hs))

    return(H)

def convert_time_to_frequency(num_samples_data, num_samples_prefix, signal_time):
    """Convert a received OFDM signal from the time domain to the frequency.

    This function handles removing the cyclic prefixes from the time domain
    signal before taking the IDFT of the packets.

    Args:
        num_samples_data (int): The number of samples of data in each packet.
        num_samples_prefix (int): The number of samples to use as the cyclic
            prefix of each packet.
        signal_time (1D float32 ndarray): The bit sequence in the time domain.

    Returns:
        signal_freq: A frequency domain signal containing packets of data.
    """
    num_samples_packet = num_samples_data + num_samples_prefix
    num_packets = signal_time.shape[-1] // num_samples_packet
    
    num_samples_signal_freq = num_samples_data * num_packets
    signal_freq = np.zeros(num_samples_signal_freq, dtype=np.complex64)

    for i in range(num_packets):
        data_start_time = num_samples_packet * i + num_samples_prefix
        data_end_time = data_start_time + num_samples_data
        
        data_start_freq = num_samples_data * i
        data_end_freq = data_start_freq + num_samples_data

        signal_freq[data_start_freq:data_end_freq] = signal_time[data_start_time:data_end_time]

    return signal_freq

def equalize_frequency(channel_estimation, signal_freq):
    """Correct a frequency domain signal for the effects of a flat fading channel.

    Args:
        channel_estimation (complex float): The flat-fading channel estimation
            in the frequency domain.
        signal_freq: A frequency domain signal containing packets of data.

    Returns:
        signal_freq_eq: An equalized frequency domain signal.
    """
    signal_freq_eq = signal_freq / channel_estimation
    return signal_freq_eq

def decode_signal_freq(signal_freq):
    """Decode a frequency domain signal into a series of bits.

    Args:
        signal_freq (1D float32 ndarray): The bit sequence in the frequency
            domain.

    Returns:
        bits (1D ndarray): The bit sequence decoded from signal_freq.
    """
    bits = np.sign(signal_freq)
    return bits

def calculate_error(bits_tx, bits_est):
    """Calculate the percent bit error between a transmitted and estimated bit sequence.

    Args:
        bits_tx (1D ndarray): A sequence of 1s and -1s representing the transmitted symbols.
        bits_est (1D ndarray): A sequence of 1s and -1s representing the
            estimated symbols.

    Returns:
        percent_error (float): The percent error of the estimated bits vs the
            transmitted bits.
    """
    assert bits_tx.shape == bits_est.shape

    num_wrong = np.sum(bits_tx != bits_est)
    percent_error = 100 * num_wrong / bits_tx.shape[-1]

    return percent_error
