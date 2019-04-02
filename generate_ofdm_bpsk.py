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

# Create the data to transmit.
seed = 10
data_freq_tx = ofdm.create_signal_freq_domain(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_PACKETS, seed)
data_time_tx = ofdm.create_signal_time_domain(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, data_freq_tx)

# Concatenate the LTS, channel estimation signal, and the data together and transmit.
signal_time_tx = np.concatenate((lts, known_signal_time_tx, data_time_tx))

# Interleave real and imaginary samples to transmit with USRP.
tmp = np.zeros(2 * signal_time_tx.shape[-1], dtype=np.float32)
tmp[::2] = signal_time_tx.real
tmp[1::2] = signal_time_tx.imag

# Save to a binary file.
tmp.tofile('tx_bpsk.dat')
