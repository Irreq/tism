#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# src/modulate.py

# Author : Irreq

"""
DOCUMENTATION:          Modulation of string or binary sequence to
                        amplitude modulated signal.

TODO:                   Add more modulation schemes and comment
                        the program for easier readability.
"""

import numpy as np

from .encoding import str_to_bin



# =============== Start Modulation =======================

class Modulation(object):

    def __init__(self, frequency=1e3, samplingrate=44.1e3,
                        bitrate=10, amplitude=1.0, encoding=True):

        """
        Class for data modulation.

        NOTE:                    At the moment, only amplitude modulation
                                 with manchester code is availible.

        ARGUMENTS:
                                 None

        KEYWORD ARGUMENTS:
            - frequency:         int() or float() representing carrier
                                 frequency in Hz. Eg, 1e3 (1000Hz)

            - samplingrate:      int() or float() representing the
                                 samplingrate for the modulation scheme.
                                 Eg, 44.1e3 (44100Hz, CD quality)

            - bitrate:           int() or float() representing transfer
                                 speed of bits per second. - meaning
                                 how many ones or zeros are in one second.
                                 Eg, 10

            - amplitude:         int() or float() representing amplitude
                                 of signal. Integers are discouraged as
                                 1 represent max value. If amplitude is
                                 greater, signal loss or other errors will
                                 occur in the physical world. Amplitude is
                                 usually 0 <= amplitude <= 1. Eg, 0.9

            - encoding:          str() or Boolean() representing the
                                 encoding scheme that will be used for
                                 the modulation. If encoding is set to True
                                 scheme will be chosen automatically.
                                 Eg, "manchester"

        TODO:                    None
        """

        assert float(frequency) > 0
        self.frequency = frequency

        assert float(samplingrate) > 2 * frequency, "Insufficient Nyquist rate"
        self.samplingrate = samplingrate

        assert float(bitrate) > 0
        self.bitrate = bitrate

        assert 0 <= amplitude <= 1, "Amplitude is out of boundaries."
        self.amplitude = amplitude

        self.encoding = encoding


    def sine_wave_generator(self, duration):

        """
        Generate a sinusoidal wave.

        Note:                    This function can be used for any type of
                                 sinusoidal-like signal, and not exclusively
                                 amplitude modulation.

        ARGUMENTS:
            - duration:          float() or int() representing duration in
                                 seconds. Eg, 2.42

        RETURNS:
            - sinusoidal_wave:   numpy.array() the created sinusoidal wave.

        TODO:                    Fix that other types of inputs can be used,
                                 and not only in seconds.
        """

        # Generates from 0 to 1/Fbit with steps from fs

        timesteps = np.arange(0, duration, 1/self.samplingrate, dtype=np.float)

        # Carrier wave

        sinusoidal_wave = np.sin(2 * np.pi * self.frequency * timesteps)

        return sinusoidal_wave

    def smooth(self, data, n, curve=0.05):

        """
        Make smooth transitions between numbers in a sequence.

        NOTE:                   This function is used for generate a smooth
                                transition between the binary integers in the
                                incoming data such that the loudspeaker will
                                have more smoother envelope without abrubt
                                changes. However this function needs to create
                                data. Eg, [0 1 0] with multiplication 2 becomes:
                                [0 0.2 0.8 1 0.8 0.2 0]

        ARGUMENTS:
            - data:             list() or numpy.array() (Usually)
                                The sequence that will be manipulated.
                                Eg, "Hello, World!"

            - n:                int() multiplication factor representing
                                number of resolution.
                                Eg, 100

        KEYWORD ARGUMENTS:
            -curve:             float() the smoothness of the change
                                a higher value result in a flatter change.
                                Eg, 0.05

        RETURNS:
            - smoothed:         list() The smoothed out sequnce.

        TODO:                   The function uses some werid parsing when
                                multiplying the values, thus it has to be split
                                as follows. This is a low priority bug.
                                Ease readability by writing more "pythonic".
        """

        # Bug below |V|

        data = [data[0], *data, data[-1]]

        D = np.linspace(0,2, n)

        sigmaD = 1 / (1 + np.exp(-(1 - D) / curve))

        def sigma(x0, x1):

            return x0 + (x1 - x0)*(1 - sigmaD)

        smoothed = [c for i in range(len(data)) if i+1 < len(data) for c in sigma(data[i],data[i+1])]

        start, end = int(np.floor(n/2)), int(np.ceil(n/2))

        return smoothed[start:-end]


    def modulate(self, payload):

        """
        Modulate data into a signal.

        NOTE:                   At the moment, only amplitude
                                modulation is availible.

        ARGUMENTS:
            - payload:          str() or list() or numpy.array()
                                this is the data that will be
                                modulated.

        RETURNS:
            - carrier:          numpy.array() signal containing the
                                modulated payload.

        TODO:                   Fix more modulation schemes and ease
                                readability by writing more "pythonic".
        """

        # Lower the amplitude

        if type(payload).__name__ == "str":

            payload = str_to_bin(payload, encoding=self.encoding)

            payload = [int(i) for i in payload]


        # data preprocessing

        A = self.amplitude

        payload = [i if i==A else A*0.5 for i in payload]

        bit_length = int(self.samplingrate / self.bitrate)

        pre_modulated_signal = np.array(self.smooth(payload, bit_length))

        duration = len(pre_modulated_signal) / self.samplingrate

        carrier = self.sine_wave_generator(duration)

        carrier *= pre_modulated_signal

        carrier /= max(abs(carrier))

        return carrier

# ================= End Modulation =======================
