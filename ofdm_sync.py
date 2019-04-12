"""This script contains code to implement a simulated OFDM system, as described
from utils import ofdm_util as ofdm
in Lab 3 part a. The signals in this script are timing synchronized.

Author: Ariana Olson

Usage:
    python3 ofdm_sync.py
"""
from __future__ import print_function, division

from utils import ofdm_util as ofdm
from utils import nonflat_channel

import matplotlib.pyplot as plt
import numpy as np

# Create a short known signal to use for channel estimation.
seed = 5
num_known_packets = 100

known_signal_freq_tx = ofdm.create_signal_freq_domain(
        ofdm.NUM_SAMPLES_PER_PACKET,
        num_known_packets,
        seed)

known_signal_time_tx = ofdm.create_signal_time_domain(
        ofdm.NUM_SAMPLES_PER_PACKET,
        ofdm.NUM_SAMPLES_CYCLIC_PREFIX,
        known_signal_freq_tx)


# Transmit known signal through the channel.
known_signal_time_rx = nonflat_channel.nonflat_channel(known_signal_time_tx)
known_signal_time_rx = ofdm.detect_start(known_signal_time_rx, known_signal_time_tx, known_signal_time_tx.shape[-1])
known_signal_freq_rx = ofdm.convert_time_to_frequency(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, known_signal_time_rx)

# Estimate the channel using the known signals.
H = ofdm.estimate_channel(known_signal_freq_tx, known_signal_freq_rx)

# Create a signal to transmit.
seed = 10
signal_freq_tx = ofdm.create_signal_freq_domain(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_PACKETS, seed)

# Take the first two data packets to be the known header.
header_length = ofdm.NUM_HEADER_PACKETS * ofdm.NUM_SAMPLES_PER_PACKET
header_freq = signal_freq_tx[:header_length]

signal_time_tx = ofdm.create_signal_time_domain(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, signal_freq_tx) / ofdm.NUM_SAMPLES_PER_PACKET
header_time = ofdm.create_signal_time_domain(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, header_freq)

# Plot the frequency and time domain signals.
plt.figure(figsize=(10, 10))
plt.suptitle("Transmitted Signals")

plt.subplot(2, 1, 1)
plt.stem(signal_freq_tx[:ofdm.NUM_SAMPLES_PER_PACKET])
plt.title("Frequency Domain (first packet of {} samples)".format(ofdm.NUM_SAMPLES_PER_PACKET))

plt.subplot(2, 1, 2)
plt.plot(signal_time_tx[:ofdm.NUM_SAMPLES_PER_PACKET + ofdm.NUM_SAMPLES_CYCLIC_PREFIX])
plt.title("Time Domain with Cyclic Prefixes (first packet of {} samples with {} cyclic prefix samples".format(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX))

plt.savefig('figures/tx_ofdm_sync.png')
plt.show()

# Transmit signal through the channel.
signal_time_rx = nonflat_channel.nonflat_channel(signal_time_tx)

signal_time_rx = ofdm.detect_start(signal_time_rx, header_time, signal_time_tx.shape[-1])

# Convert from time to frequency domain.
signal_freq_rx = ofdm.convert_time_to_frequency(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, signal_time_rx)

# Plot the time domain signal before start of the signal.
plt.figure(figsize=(10, 10))
plt.suptitle("Received Signals")

# Plot the time domain signal after finding the start.
plt.subplot(2, 1, 1)
plt.plot(signal_time_rx[:ofdm.NUM_SAMPLES_PER_PACKET + ofdm.NUM_SAMPLES_CYCLIC_PREFIX])
plt.title("Time Domain with Cyclic Prefixes After Finding Start (first packet of {} samples with {} cyclic prefix samples".format(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX))

# Plot the frequency domain signal.
plt.subplot(2, 1, 2)
plt.stem(signal_freq_rx[:ofdm.NUM_SAMPLES_PER_PACKET])
plt.title("Frequency Domain (first packet of {} samples)".format(ofdm.NUM_SAMPLES_PER_PACKET))

plt.savefig('figures/rx_ofdm_sync.png')
plt.show()

# Equalize the signal
signal_freq_eq = ofdm.equalize_frequency(H, signal_freq_rx)

plt.figure(figsize=(10, 10))
plt.suptitle("Compare Transmitted and Received and Equalized Frequency Domain Signals")

plt.subplot(2, 1, 1)
plt.stem(signal_freq_tx[:ofdm.NUM_SAMPLES_PER_PACKET])
plt.title("Transmitted Frequency Domain (first packet of {} samples)".format(ofdm.NUM_SAMPLES_PER_PACKET))

plt.subplot(2, 1, 2)
plt.stem(signal_freq_eq[:ofdm.NUM_SAMPLES_PER_PACKET])
plt.title("Received Equalized Frequency Domain (first packet of {} samples)".format(ofdm.NUM_SAMPLES_PER_PACKET))

plt.savefig('figures/compare_freq_ofdm_sync.png')
plt.show()

# Decode the signal in the frequency domain.
bits = ofdm.decode_signal_freq(signal_freq_eq)

# Calculate the percent error rate.
percent_error = ofdm.calculate_error(ofdm.decode_signal_freq(signal_freq_tx), bits)

print(np.sum(ofdm.decode_signal_freq(signal_freq_tx) == bits))
print("The bit error rate is: {}%".format(percent_error))
