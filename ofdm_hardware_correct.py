"""This script contains code to decode a received OFDM signal.

The signals we used to test this algorithm are generated in
generate_ofdm.py (BPSK signals). The signals were transmitted using Universal
Sofware Radio Peripherals (USRPs) with the following settings:
    - Samp Rate (Sps): 6e6
    - Ch0Center Freq (Hz): 2.454e9
    - Tx Gain (dB): 80

Please note that the gain settings are dependent on the environment and may
need to be changed for best results.

Authors: Annie Ku and Ariana Olson

Usage:
    - Generate and OFDM signal with generate_ofdm.py. See the documentation in
      these file for details. A binary file that can be transmitted over USRP
      is generated, along with a .npz file containing signal components in
      numpy arrays that are used in this file for decoding.
    - Save the received data to a known location.
    - Run this script with the path to the received data specified. The default
      path is 'Data/ofdmReceiveFile_70.dat'.
        python3 ofdm_hardaware_correct.py --rx_data_path=<path/to/received>.dat --tx_npz_path=<path/to/npz>.npz
    - If you would like to generate plots, please add --make_plots=True argument.
"""
from __future__ import print_function, division

from utils import ofdm_util as ofdm
from utils import nonflat_channel_timing_error

import matplotlib.pyplot as plt
import numpy as np

import argparse

# Set up commandline arguments.
FLAGS = None

parser = argparse.ArgumentParser()
parser.add_argument('--rx_data_path', type=str,
        default='Data/ofdmReceiveFile_70', help=
        'The path to the binary file of received data')
parser.add_argument('--tx_npz_path', type=str, default='tx_arrays.npz',
        help='Path to .npz file containing components of the transmitted'
        + 'signal.')
parser.add_argument('--make_plots', type=bool, default=False,
        help='Generate and display plots.')
parser.add_argument('--qpsk', type=bool, default=False,
        help='Decode a qpsk signal')

FLAGS, unparsed = parser.parse_known_args()

received_data = np.fromfile("Data/ofdmReceiveFile_70.dat", dtype=np.float32)
signal_time_rx = received_data[::2] + received_data[1::2]*1j

if FLAGS.make_plots:
    plt.plot(signal_time_rx)
    plt.title("Received Signal Without Correction")
    plt.show()

# Load signal components for decoding and calculating bit-error rates.
tx_arrays = np.load(FLAGS.tx_npz_path)
lts = tx_arrays['lts']
header_time = tx_arrays['header_time']
data_time = tx_arrays['data_time']
header_freq = tx_arrays['header_freq']
data_freq = tx_arrays['data_freq']

if FLAGS.make_plots:
    tmp = signal_time_rx
    plt.plot(signal_time_rx)

# Find the start of the data using the LTS.
signal_time_len = lts.shape[-1] + header_time.shape[-1] + data_time.shape[-1]
lag, signal_time_rx = ofdm.detect_start_lts(signal_time_rx, lts,
        signal_time_len)

if FLAGS.make_plots:
    tmp[:lag] = 0
    tmp[lag + signal_time_len:] = 0

    plt.plot(tmp)
    plt.title("received data")
    plt.show()

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
header_eq = ofdm.equalize_frequency(H, header_freq_rx, qpsk=FLAGS.qpsk, est_phase=True)

# See what the bit-error rate is for the decoded known header.
print((ofdm.decode_signal_freq(header_eq, qpsk=FLAGS.qpsk) == header_freq).mean())

# Convert from time to frequency domain.
data_time_rx = signal_time_rx[channel_est_end:]
data_freq_rx = ofdm.convert_time_to_frequency(ofdm.NUM_SAMPLES_PER_PACKET, ofdm.NUM_SAMPLES_CYCLIC_PREFIX, data_time_rx)

if FLAGS.make_plots:
    # Plot the time domain signal after finding the start.
    plt.subplot(2, 1, 1)
    plt.plot(data_time_rx[:ofdm.NUM_SAMPLES_PER_PACKET + ofdm.NUM_SAMPLES_CYCLIC_PREFIX])
    plt.title("Time Domain with Cyclic Prefixes After Finding Start (first" + 
            "packet of {} samples with {} cyclic prefix" + 
            "samples".format(ofdm.NUM_SAMPLES_PER_PACKET,
            ofdm.NUM_SAMPLES_CYCLIC_PREFIX))
    plt.plot()
    # Plot the frequency domain signal.
    plt.subplot(2, 1, 2)
    plt.stem(data_freq_rx[:ofdm.NUM_SAMPLES_PER_PACKET])
    plt.title("Frequency Domain (first packet of {}" +
            "samples)".format(ofdm.NUM_SAMPLES_PER_PACKET))
    plt.show()

# Correct for the channel and the phase offset.
data_freq_eq = ofdm.equalize_frequency(H, data_freq_rx, qpsk=FLAGS.qpsk, est_phase=False)

if FLAGS.make_plots:
    tmp = data_freq_eq[12::ofdm.NUM_SAMPLES_PER_PACKET]

    plt.plot(tmp.real, tmp.imag, ".")
    plt.title("Constellation plot of subcarrier 12")
    plt.show()

    plt.plot(data_freq_eq)
    plt.title("Received Data (Frequency) after equalization, before quantization")
    plt.show()

# Decode the signal in the frequency domain.
bits = ofdm.decode_signal_freq(data_freq_eq, qpsk=FLAGS.qpsk)

# Calculate the percent error rate.
print(data_freq.shape)
percent_error = ofdm.calculate_error(ofdm.decode_signal_freq(data_freq, qpsk=FLAGS.qpsk)[:4000], bits[:4000])

if FLAGS.make_plots:
    plt.plot(np.sign(data_freq) == bits, 'o')
    plt.show()

print("The bit error rate is: {}%".format(percent_error))
