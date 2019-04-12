# Wireless Lab 3
## Part a - OFDM with synchronized clocks and simulated channel
The script to run for part a of the lab is in `ofdm_sync.py`

This is a single self-contained script. To use, run the following command:

```
python3 ofdm_sync.py
```
## Part b - OFDM with non-synchronized clocks and simulated channel
The script to run for part b of the lab is in `ofdm_non_sync.py`.

This is a self-contained script. To use, run the following command:

```
python3 ofdm_non_sync.py
```

## Part c - OFDM with hardware and non-synchronized clocks.
This part requires access to two Universal Software Radio Peripherals.
To generate BPSK data for transmission run the following command:

```
python3 generate_ofdm.py
```

To generate QPSK data for transmission run the following command:
```
python3 generate_ofdm.py --use_qpsk=True
```

This data must then be transmitted using the USRPs. We used GNU radio companion to do this. Save the received data in a know location.

To decode the received data:

```
python ofdm_hardware_correct.py --rx_data_path=<path/to/received/data>
```

*Please see the documentation in each script for descriptions of all command line options.*
