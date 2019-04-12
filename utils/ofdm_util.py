"""This file contains functions to implement OFDM for Lab 3.

Author: Ariana Olson
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

# Define constants used in this implementation.
NUM_SAMPLES_PER_PACKET = 64
NUM_PACKETS = 1000
NUM_HEADER_PACKETS = 10
NUM_SAMPLES_CYCLIC_PREFIX = 16

def create_signal_freq_domain(num_samples, num_packets, seed, pilot=False, qpsk=False):
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
    signal_freq_real = np.sign(np.random.randn(num_samples * num_packets))

    if qpsk:
        np.random.seed(seed * 2)
        signal_freq_imag = np.sign(np.random.randn(num_samples * num_packets))
    else:
        signal_freq_imag = np.zeros(signal_freq_real.shape)

    signal_freq = signal_freq_real + 1j * signal_freq_imag

    if pilot:
        if qpsk:
            pilot7 = 1 + 1j
            pilot21 = 1 + 1j
            pilot44 = 1 + 1j
            pilot58 = 1 + 1j
        else:
            pilot7 = 1
            pilot21 = 1
            pilot44 = 1
            pilot58 = 1

        for i in range(0, signal_freq.shape[-1], num_samples):
            signal_freq[i + 7] = pilot7
            signal_freq[i + 21] = pilot21
            signal_freq[i + 44] = pilot44
            signal_freq[i + 58] = pilot58

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
    num_time_samples_per_packet = num_samples_data + num_samples_prefix
    num_time_samples = num_time_samples_per_packet  * num_packets

    signal_time = np.zeros(num_time_samples, dtype=np.complex64)

    # Create data packets with cyclic prefixes.
    for i in range(num_packets):
        packet_start = i * num_samples_data
        packet_end = packet_start + num_samples_data

        time_data_packet = np.fft.ifft(signal_freq[packet_start:packet_end])
        prefix = time_data_packet[-num_samples_prefix:]

        prefix_start = i * num_time_samples_per_packet
        prefix_end = prefix_start + num_samples_prefix

        data_start = prefix_end
        data_end = prefix_end + num_samples_data
        
        # Add prefix
        signal_time[prefix_start:prefix_end] = prefix

        # Add data after prefix
        signal_time[data_start:data_end] = time_data_packet

    return signal_time

def detect_start(signal_time_rx, header, signal_length):
    """Find the start of the time domain signal using crosss correlation.

    Args:
        signal_time_rx (1D complex ndarray): A received time domain signal
            that conatains a known header as the first portion of the signal.
        header (1D complex ndarray): A known signal that is included at the
            beginning of the transmitted signal.
        signal_length (int): The length of the signal that whas transmitted.

    Returns:
        signal_time_start (1D complex ndarray): A version of signal_time_rx
            that starts at the first data sample.
    """
    cross_corr = np.correlate(signal_time_rx, header)
    lag = np.argmax(cross_corr) - 1

    return signal_time_rx[lag:lag+signal_length]

def detect_start_lts(signal_time_rx, lts, signal_length):
    cross_corr = np.correlate(signal_time_rx, lts)
    lag = np.argmax(np.abs(cross_corr)) - 1

    return lag, signal_time_rx[lag:lag+signal_length]

def estimate_channel(tx_known_signal_frequency, rx_known_signal_frequncy, make_plots=False):
    """Estimate the frequency domain channel coefficient from a set of transmitted and received known signals.

    Args:
        tx_known_signals_frequency (1D ndarray): Frequency domain known signal transmitted through the channel.
        rx_known_signals_frequency (1D ndarray): Frequency domain received signal corresponding to tx_known_signal_frequency.
        make_plots (bool): Make and show a visualization of the coefficeient values. Defaults to False.

    Returns:
        H (complex float): the estimated channel in the frequency domain.
    """
    assert tx_known_signal_frequency.shape == rx_known_signal_frequncy.shape

    H = rx_known_signal_frequncy / tx_known_signal_frequency
    H = H.reshape(rx_known_signal_frequncy.shape[-1] // NUM_SAMPLES_PER_PACKET, NUM_SAMPLES_PER_PACKET)

    H = np.mean(H, axis=0)

    if make_plots:
        plt.plot(H)
        plt.title("Coefficient Values")
        plt.show()
    return(H)

def estimate_phase(packet_frequency, qpsk=False):
    """Estimate the average phase off set of a packet of ofdm data using pilot bits.

    Args: 
        packet_frequency (ndarray): An array representing a single packet of data in the frequency domain.
        qpsk (bool): Specifies that the signal is a QPSK signal instead of a
            BPSK signal. Defaults to False.

    Returns:
        phase (complex float): The estimated average phase offset.
    """
    if qpsk:
        phase7 = np.angle(packet_frequency[7]/ (1 + 1j))
        phase21 = np.angle(packet_frequency[21]/ (1 + 1j))
        phase44 = np.angle(packet_frequency[44]/ (1 + 1j))
        phase58 = np.angle(packet_frequency[58]/ (1 + 1j))

    else:
        phase7 = np.angle(packet_frequency[7]/ (1))
        phase21 = np.angle(packet_frequency[21]/ (1))
        phase44 = np.angle(packet_frequency[44]/ (1))
        phase58 = np.angle(packet_frequency[58]/ (1))

    return (phase7 + phase21 + phase44 + phase58) / 4

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

        signal_freq[data_start_freq:data_end_freq] = np.fft.fft(signal_time[data_start_time:data_end_time])

    return signal_freq

def equalize_frequency(channel_estimation, signal_freq, est_phase=False, qpsk=False, make_plots=False):
    """Correct a frequency domain signal for the effects of a flat fading channel.

    Args:
        channel_estimation (complex float): The flat-fading channel estimation
            in the frequency domain.
        signal_freq (1D complex ndarray): A frequency domain signal containing
            packets of data.
        est_phase (bool): Estimate the phase offset from pilot bits. Defaults to False.
        qpsk (bool): Specifies that the signal is a QPSK signal instead of a
            BPSK signal. Defaults to False.
        make_plots (bool): Make and display a visualization of the phase
            estimates. Defaults to False. This only works if est_phase is also
            true.

    Returns:
        signal_freq_eq (1D complex ndarray): An equalized frequency domain signal.
    """
    assert signal_freq.shape[-1] % channel_estimation.shape[-1] == 0
    phase_estimates = []

    for i in range(0, signal_freq.shape[-1], NUM_SAMPLES_PER_PACKET):
        signal_freq[i:i+NUM_SAMPLES_PER_PACKET] = signal_freq[i:i+NUM_SAMPLES_PER_PACKET] / channel_estimation
        if est_phase:
            phase_est = estimate_phase(signal_freq[i:i+NUM_SAMPLES_PER_PACKET], qpsk=False)
            phase_estimates.append(phase_est)
            signal_freq[i:i+NUM_SAMPLES_PER_PACKET] /= np.exp(1j*phase_est)
    
    if make_plots and est_phase:
        plt.plot(phase_estimates)
        plt.title("phase estimates")
        plt.show()
    return signal_freq

def decode_signal_freq(signal_freq):
    """Decode a frequency domain signal into a series of bits.

    Args:
        signal_freq (1D float32 ndarray): The bit sequence in the frequency
            domain.

    Returns:
        bits (1D ndarray): The bit sequence decoded from signal_freq.
    """
    bits = np.sign(signal_freq.real) + 1j * np.sign(signal_freq.imag)

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

    num_wrong = 0

    for cycle_i in range(NUM_PACKETS):
        index = cycle_i * NUM_SAMPLES_PER_PACKET
        bits_tx_block1 = bits_tx[index + 1:index + 27]
        bits_est_block1 = bits_est[index + 1:index + 27]
        bits_tx_block2 = bits_tx[index + 38:index + 64]
        bits_est_block2 = bits_est[index + 38: index + 64]
        num_wrong += np.sum(bits_tx_block1 != bits_est_block1) + np.sum(bits_tx_block2 != bits_est_block2)
    # num_wrong = np.sum(bits_tx != bits_est)
    print('summed wrong', num_wrong)
    percent_error = 100 * num_wrong / (NUM_PACKETS * (26 + 27))# bits_tx.shape[-1]

    return percent_error

def create_long_training_sequence(num_samples, seed_real, seed_imag):
    """Create one block of the long training sequence used by the Schmidl Cox
    Algorithm.

    Args:

        num_samples: The number of samples in a block.
        seed_real: The seed used for the random number generator to generate
            the real portion of the signal.
        seed_imag: The seed used for the random number generator to generate
            the imag portion of the signal.

    Returns

        lts (1D complex ndarray): A block of random complex values taking on
            values of +-1 +-j of length 3 * num_samples. This contains 3 repeating
            blocks of size num_samples.
    """
    np.random.seed(seed_real)
    reals = np.sign(np.random.randn(num_samples))

    np.random.seed(seed_imag)
    imags = np.sign(np.random.randn(num_samples))
    
    lts_block = reals + 1j * imags
    lts = np.zeros(num_samples * 3, dtype=np.complex64)
    lts[:num_samples] = lts_block
    lts[num_samples:2 * num_samples] = lts_block
    lts[2 * num_samples:3 * num_samples] = lts_block

    return lts

def estimate_f_delta(lts, num_samples):
    """Estimate the frequency offset of a received OFDM signal using the LTS.

    Args:
        lts (1D complex ndarray): A block of random complex values taking on
            values of +-1 +-j of length 3 * num_samples that has been sent through a nonflat channel. This contains 3 repeating
            blocks of size num_samples.
        num_samples (int): The number of samples in each LTS block.

    Returns:
        f_delta_est (float): The average estimated frequency offset, in radians.
    """
    sum_f_delta_ests = 0
    for i in range(num_samples):
        complex_exp = lts[2 * num_samples + i] / lts[num_samples + i]
        sum_f_delta_ests += np.angle(complex_exp)

    return sum_f_delta_ests / (num_samples ** 2)

def correct_freq_offset(signal_time, f_delta):
    """Correct for the frequency offset in a time-domain signal.

    Args:
        signal_time (1D complex ndarray): A time domain signal that starts with
            the LTS used to estimate the frequency offset.
        f_delta (float): The frequency offset to correct for, in radians.

    Returns:
        signal_time_corrected (1D complex ndarray): The corrected time domain
            signal. Has the same shape as signal_time.
    """
    exponentials = np.exp(np.arange(signal_time.shape[-1]) * 1j * f_delta)
    signal_time_corrected = signal_time / exponentials

    return signal_time_corrected
