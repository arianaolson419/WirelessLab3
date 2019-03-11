"""This script contains code to implement a simulated OFDM system, as described
in Lab 3 part a. The signals in this script are timing synchronized.

Author: Ariana Olson
"""
from __future__ import print_function, division

from utils import ofdm_util
from utils import nonflat_channel.py

# Create a signal to transmit.
# Create a short known signal to use for channel estimation.
# Transmit signal through the channel.
# Estimate the channel with the known signal.
# Convert from time to frequency domain.
# Equalize the signal
# decode the signal in the frequency domain.
