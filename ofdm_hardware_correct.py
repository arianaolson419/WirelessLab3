from __future__ import print_function, division

from utils import ofdm_util as ofdm
from utils import nonflat_channel_timing_error

import matplotlib.pyplot as plt
import numpy as np

# TODO: read in received data from file.
received_data = np.fromfile("Data/ofdmReceiveFile_70.dat", dtype=np.float32)
# TODO: make interleaved signal into a complex signal
signal_time_rx = received_data[::2] + received_data[1::2]*1j

# TODO: load in saved lts, header, data. 
tx_arrays = np.load('tx_arrays.npz')
lts = tx_arrays['lts']
header_time = tx_arrays['header']
data_time = tx_arrays['data']


# Find the start of the data using the LTS.
signal_time_rx = ofdm.detect_start_lts(signal_time_rx, lts, signal_time_tx.shape[-1])

# Estmate f_delta using the LTS.
lts_rx = signal_time_rx[:lts.shape[-1]]
f_delta_est = ofdm.estimate_f_delta(lts_rx, ofdm.NUM_SAMPLES_PER_PACKET)
print(f_delta_est, 'my f_delta')

# Correct for f_delta.
signal_time_rx = ofdm.correct_freq_offset(signal_time_rx, f_delta_est)

# Estimate the channel using the known channel estimation sequence.
channel_est_start = lts.shape[-1]
channel_est_end = channel_est_start + known_signal_time_tx.shape[-1]

known_signal_time_rx = signal_time_rx[channel_est_start:channel_est_end]
known_signal_freq_rx = ofdm.convert_time_to_frequency(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, known_signal_time_rx)

H = ofdm.estimate_channel(known_signal_freq_tx, known_signal_freq_rx)
known_signal_eq = ofdm.equalize_frequency(H, known_signal_freq_rx)

# See what the bit-error rate is for the decoded known header.
print((ofdm.decode_signal_freq(known_signal_eq) == known_signal_freq_tx).mean())

# Convert from time to frequency domain.
data_time_rx = signal_time_rx[channel_est_end:]
data_freq_rx = ofdm.convert_time_to_frequency(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, data_time_rx)

# Equalize the signal
data_freq_eq = ofdm.equalize_frequency(H, data_freq_rx)

# Decode the signal in the frequency domain.
bits = ofdm.decode_signal_freq(data_freq_eq)

# Calculate the percent error rate.
percent_error = ofdm.calculate_error(np.sign(data_freq_tx), bits)

print("The bit error rate is: {}%".format(percent_error))
