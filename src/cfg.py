#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# src/buffer.py

# Author : Irreq

"""
DOCUMENTATION:          Global variables are being stored here.

TODO:                   Fix structure and ease readability and
                        optimize for performance.
"""

buffer = {}

status = False

stream_to_modulate = []

buffer_1 = []

buffer_2 = []

demodulated_stream = []

# Cannot be numpy array must be extended list
stream_to_send = []

# the in and outbound streams
streams = []




args = {
        'scheme' : {
                      '0':[0],
                     '00':[0,0],
                      '1':[1],
                     '11':[1,1],
                    },
                    









}
