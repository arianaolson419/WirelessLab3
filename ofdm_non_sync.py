"""This script contains code to implement OFDM for a simulated signal that is
not timimg synchronizezd. Ti=he Schmidl-Cox algortithm is used to perform
timing synchronization.
"""
from __future__ import print_function, division

from utils import ofdm_util as ofdm
from utils import nonflat_channel_timing_error

import matplotlib.pyplot as plt
import numpy as np

# Create the long training sequence (LTS). This is a random sequence of 64
# complex time domain samples.
seed_real = 5
seed_imag = 3
lts = ofdm.create_long_training_sequence(ofdm.NUM_SAMPLES_PER_PACKET, seed_real, seed_imag)

# Create the channel estimation sequence.
seed = 11
num_known_packets = 100

known_signal_freq_tx = ofdm.create_signal_freq_domain(
        ofdm.NUM_SAMPLES_PER_PACKET,
        num_known_packets,
        seed)

known_signal_time_tx = ofdm.create_signal_time_domain(
        ofdm.NUM_SAMPLES_PER_PACKET,
        ofdm.NUM_SAMPLES_CYCLIC_PREFIX,
        known_signal_freq_tx)
plt.subplot(2, 1, 1)
plt.plot(known_signal_time_tx)

# Create the data to transmit.
seed = 10
data_freq_tx = ofdm.create_signal_freq_domain(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_PACKETS, seed)
data_time_tx = ofdm.create_signal_time_domain(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, data_freq_tx) / ofdm.NUM_SAMPLES_PER_PACKET

plt.subplot(2, 1, 2)
plt.plot(data_time_tx)
plt.show()

# Concatenate the LTS, channel estimation signal, and the data together and transmit.
signal_time_tx = np.concatenate((lts, known_signal_time_tx, data_time_tx))

# Transmit signal through the channel.
signal_time_rx = nonflat_channel_timing_error.nonflat_channel(signal_time_tx)
tmp = np.copy(signal_time_rx)
tmp[:80] = 0
plt.plot(signal_time_rx)
plt.plot(tmp)
plt.show()

# Find the start of the data using the LTS.
signal_time_rx = ofdm.detect_start_lts(signal_time_rx, lts, signal_time_tx.shape[-1])
print('short')
print(signal_time_rx.shape, signal_time_tx.shape)

# Estmate f_delta using the LTS.
lts_rx = signal_time_rx[:lts.shape[-1]]
f_delta_est = ofdm.estimate_f_delta(lts_rx, ofdm.NUM_SAMPLES_PER_PACKET)

# Correct for f_delta.
signal_time_rx = ofdm.correct_freq_offset(signal_time_rx, f_delta_est)
print(signal_time_rx.shape, 'after freq offset correction')

# Estimate the channel using the known channel estimation sequence.
channel_est_start = lts.shape[-1]
channel_est_end = channel_est_start + known_signal_time_tx.shape[-1]

known_signal_time_rx = signal_time_rx[channel_est_start:channel_est_end]
known_signal_freq_rx = ofdm.convert_time_to_frequency(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, known_signal_time_rx)

H = ofdm.estimate_channel([known_signal_freq_tx], [known_signal_freq_rx])

# Convert from time to frequency domain.
data_time_rx = signal_time_rx[channel_est_end:]
data_freq_rx = ofdm.convert_time_to_frequency(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, data_time_rx)

# Equalize the signal
data_freq_rx = ofdm.equalize_frequency(H, data_freq_rx)

## Decode the signal in the frequency domain.
bits = ofdm.decode_signal_freq(data_freq_rx)
print(bits.shape, data_freq_tx.shape)

# Calculate the percent error rate.
percent_error = ofdm.calculate_error(data_freq_tx, bits)

print("The bit error rate is: {}%".format(percent_error))
