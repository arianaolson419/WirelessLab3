from __future__ import print_function, division

from utils import ofdm_util as ofdm
from utils import nonflat_channel_timing_error

import matplotlib.pyplot as plt
import numpy as np

received_data = np.fromfile("Data/ofdmReceiveFile_70.dat", dtype=np.float32)
signal_time_rx = received_data[::2] + received_data[1::2]*1j

# plt.plot(signal_time_rx)
# plt.title("Received Signal Without Correction")
# plt.show()


tx_arrays = np.load('tx_arrays.npz')
lts = tx_arrays['lts']
header_time = tx_arrays['header_time']
data_time = tx_arrays['data_time']
header_freq = tx_arrays['header_freq']
data_freq = tx_arrays['data_freq']

tmp = signal_time_rx

# plt.plot(signal_time_rx)

# Find the start of the data using the LTS.
signal_time_len = lts.shape[-1] + header_time.shape[-1] + data_time.shape[-1]
lag, signal_time_rx = ofdm.detect_start_lts(signal_time_rx, lts, signal_time_len)
tmp[:lag] = 0
tmp[lag + signal_time_len:] = 0

# plt.plot(tmp)
# plt.title("received data")
# plt.show()

# Estmate f_delta using the LTS.
lts_rx = signal_time_rx[:lts.shape[-1]]
f_delta_est = ofdm.estimate_f_delta(lts_rx, ofdm.NUM_SAMPLES_PER_PACKET)
print(f_delta_est, 'estimated f_delta')

# Correct for f_delta.
signal_time_rx = ofdm.correct_freq_offset(signal_time_rx, f_delta_est)

# Estimate the channel using the known channel estimation sequence.
channel_est_start = lts.shape[-1]
channel_est_end = channel_est_start + header_time.shape[-1]

header_time_rx = signal_time_rx[channel_est_start:channel_est_end]
header_freq_rx = ofdm.convert_time_to_frequency(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, header_time_rx)

H = ofdm.estimate_channel(header_freq, header_freq_rx)
header_eq = ofdm.equalize_frequency(H, header_freq_rx, est_phase=True)

# See what the bit-error rate is for the decoded known header.
print((ofdm.decode_signal_freq(header_eq) == header_freq).mean())

# Convert from time to frequency domain.
data_time_rx = signal_time_rx[channel_est_end:]
data_freq_rx = ofdm.convert_time_to_frequency(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, data_time_rx)



# Plot the time domain signal after finding the start.
plt.subplot(2, 1, 1)
plt.plot(data_time_rx[:ofdm.NUM_SAMPLES_PER_PACKET + ofdm.NUM_SAMPLES_CYCLIC_PREFIX])
plt.title("Time Domain with Cyclic Prefixes After Finding Start (first packet of {} samples with {} cyclic prefix samples".format(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX))
plt.plot()
# Plot the frequency domain signal.
plt.subplot(2, 1, 2)
plt.stem(data_freq_rx[:ofdm.NUM_SAMPLES_PER_PACKET])
plt.title("Frequency Domain (first packet of {} samples)".format(ofdm.NUM_SAMPLES_PER_PACKET))
plt.show()


# Correct for the channel and the phase offset.
data_freq_eq = ofdm.equalize_frequency(H, data_freq_rx, est_phase=False)

plt.plot(data_freq_eq)
plt.title("Received Data (Frequency) after equalization, before quantization")
plt.show()





# Decode the signal in the frequency domain.
bits = ofdm.decode_signal_freq(data_freq_eq)

# Calculate the percent error rate.
print(data_freq.shape)
percent_error = ofdm.calculate_error(np.sign(data_freq)[:4000], bits[:4000])
plt.plot(np.sign(data_freq) == bits, 'o')
plt.show()

print("The bit error rate is: {}%".format(percent_error))
