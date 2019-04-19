"""This script is used to generate a BPSK or QPSK OFDM signal to transmit using USRPs.

Usage:
    - Generate a BSPK signal:
        python3 generate_ofdm.py
    - Generate a QPAK signal:
        python3 generate_ofdm.py --use_qpsk=True
"""
from __future__ import print_function, division

from utils import ofdm_util as ofdm
from utils import nonflat_channel_timing_error

import matplotlib.pyplot as plt
import numpy as np

import argparse

FLAGS = None
parser = argparse.ArgumentParser()

parser.add_argument('--use_qpsk', type=bool, default=False,
        help='Specify that the generated data is QPSK instead of BPSK.')

FLAGS, unparsed = parser.parse_known_args()

# Create the long training sequence (LTS). This is a random sequence of 64
# complex time domain samples.
seed_real = 5
seed_imag = 3
lts = ofdm.create_long_training_sequence(ofdm.NUM_SAMPLES_PER_PACKET, seed_real, seed_imag)

# Create the channel estimation sequence.
seed = 11
num_known_packets = 10

known_signal_freq_tx = ofdm.create_signal_freq_domain(
        ofdm.NUM_SAMPLES_PER_PACKET,
        num_known_packets,
        seed,
        pilot=True,
        qpsk=FLAGS.use_qpsk)

known_signal_time_tx = ofdm.create_signal_time_domain(
        ofdm.NUM_SAMPLES_PER_PACKET,
        ofdm.NUM_SAMPLES_CYCLIC_PREFIX,
        known_signal_freq_tx)

# Create the data to transmit.
seed = 10
data_freq_tx = ofdm.create_signal_freq_domain(ofdm.NUM_SAMPLES_PER_PACKET,
        ofdm.NUM_PACKETS, seed, qpsk=FLAGS.use_qpsk)
data_time_tx = ofdm.create_signal_time_domain(ofdm.NUM_SAMPLES_PER_PACKET,
        ofdm.NUM_SAMPLES_CYCLIC_PREFIX, data_freq_tx)
rms = np.sqrt(np.mean(np.square(np.abs(data_time_tx))))

# Zero padding
zero_pad = np.zeros(5000)

# Concatenate the LTS, channel estimation signal, and the data together and transmit.
signal_time_tx = np.concatenate((zero_pad, lts * rms, known_signal_time_tx, data_time_tx))

# Normalize to +/- 0.5.
signal_time_tx = 0.5 * signal_time_tx / np.max(np.abs(signal_time_tx))

# Interleave real and imaginary samples to transmit with USRP.
tmp = np.zeros(2 * signal_time_tx.shape[-1], dtype=np.float32)
tmp[::2] = signal_time_tx.real
tmp[1::2] = signal_time_tx.imag

plt.plot(tmp)
plt.show()

# Save to a binary file.
tmp.tofile('tx_data.dat')

# Save the comonents of the transmitted signal for analysis and correction at the receiver.
dest = 'tx_arrays.npz'

np.savez(dest, lts=lts, header_time=known_signal_time_tx,
        data_time=data_time_tx, header_freq=known_signal_freq_tx,
        data_freq=data_freq_tx)
