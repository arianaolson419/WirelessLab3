"""This script contains code to implement a simulated OFDM system, as described
in Lab 3 part a. The signals in this script are timing synchronized.

Author: Ariana Olson
"""
from __future__ import print_function, division

from utils import ofdm_util as ofdm
from utils import nonflat_channel

import matplotlib.pyplot as plt
import numpy as np

# Create a short known signal to use for channel estimation.
seed = 5
known_signal_freq_tx = ofdm.create_signal_freq_domain(ofdm.NUM_SAMPLES_PER_PACKET, 100, seed)
known_signal_time_tx = ofdm.create_signal_time_domain(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, known_signal_freq_tx)

estimation_iterations = 1
known_signals_rx = []

# Transmit known signal through the channel.
for i in range(estimation_iterations):
    known_signal_time_rx = nonflat_channel.nonflat_channel(known_signal_time_tx)
    known_signal_time_rx = ofdm.detect_start(known_signal_time_rx, known_signal_time_tx, known_signal_time_tx.shape[-1])
    known_signals_rx.append(ofdm.convert_time_to_frequency(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, known_signal_time_rx))

# Estimate the channel using the known signals.
H = ofdm.estimate_channel([known_signal_freq_tx] * estimation_iterations, known_signals_rx)

# Create a signal to transmit.
seed = 10
signal_freq_tx = ofdm.create_signal_freq_domain(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_PACKETS, seed)

# Take the first two data packets to be the known header.
header_length = ofdm.NUM_HEADER_PACKETS * ofdm.NUM_SAMPLES_PER_PACKET
header_freq = signal_freq_tx[:header_length]

signal_time_tx = ofdm.create_signal_time_domain(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, signal_freq_tx) / ofdm.NUM_SAMPLES_PER_PACKET
header_time = ofdm.create_signal_time_domain(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, header_freq)

# Transmit signal through the channel.
signal_time_rx = nonflat_channel.nonflat_channel(signal_time_tx)

signal_time_rx = ofdm.detect_start(signal_time_rx, header_time, signal_time_tx.shape[-1])

# Convert from time to frequency domain.
signal_freq_rx = ofdm.convert_time_to_frequency(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, signal_time_rx)

# Equalize the signal
signal_freq_eq = ofdm.equalize_frequency(H, signal_freq_rx)

# Decode the signal in the frequency domain.
bits = ofdm.decode_signal_freq(signal_freq_eq)

# Calculate the percent error rate.
percent_error = ofdm.calculate_error(signal_freq_tx, bits)

print("The bit error rate is: {}%".format(percent_error))
