#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# src/modulation.py

# Author : Irreq

"""
    Modulation of string or binary list to
    amplitude modulated sinusoidal signal
"""

import numpy as np

from .encoding import str_to_bin



# =============== Start Modulation =======================

class Modulation(object):

    def __init__(self, frequency=1e3, samplingrate=44.1e3,
                        bitrate=10, amplitude=0.5, encoding=True):

        """
            frequency : the frequency of the signal
         samplingrate : the samplingrate, prefferably 44.1kHz
              bitrate : float or integer
            amplitude : prefferably <= 2.0
             encoding : Boolean


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
        Sinusoidal Wave Generating Function

            duration : second(s)

             returns : numpy.array() #sine wave
        """

        # Generates from 0 to 1/Fbit with steps from fs

        arranged = np.arange(0, duration, 1/self.samplingrate, dtype=np.float)

        # Carrier wave

        sinusoidal_wave = np.sin(2 * np.pi * self.frequency * arranged)

        return sinusoidal_wave

    def smooth(self, data, n, curve=0.05):

        """
        Frequency Change Damping Function

        Smooths signal so the speakers won't break on frequency change

           data : list
              n : the factor of multiplication
          curve : higher value results in smoother curve

        returns : list #magnified ~n times
        """

        # the function uses some werid parsing when multiplying the values,
        # thus it has to be split as follows. This is a low priority bug.

        data = [data[0], *data, data[-1]]

        D = np.linspace(0,2, n)

        sigmaD = 1 / (1 + np.exp(-(1 - D) / curve))

        def sigma(x0, x1):

            return x0 + (x1 - x0)*(1 - sigmaD)

        result = [c for i in range(len(data)) if i+1 < len(data) for c in sigma(data[i],data[i+1])]

        start, end = int(np.floor(n/2)), int(np.ceil(n/2))

        return result[start:-end]


    def modulate(self, payload):

        """
        Amplitude Modulation Function

           payload : list
          curve : higher value results in smoother curve

        returns : numpy.array() #modulated signal
        """

        if type(payload).__name__ == "str":

            payload = str_to_bin(payload, encoding=self.encoding)

            payload = [int(i) for i in payload]


        # data preprocessing

        payload = [i if i==1 else self.amplitude for i in payload]

        bit_length = int(self.samplingrate / self.bitrate)

        pre_modulated_signal = np.array(self.smooth(payload, bit_length))

        duration = len(pre_modulated_signal) / self.samplingrate

        carrier = self.sine_wave_generator(duration)

        carrier *= pre_modulated_signal

        carrier /= max(abs(carrier))

        return carrier

# ================= End Modulation =======================


def test(debug=False):

    try:
        print(" Testing modulation")

        # Test if the modulation is working

        M = Modulation(frequency=8e3, bitrate=300)

        sig = M.modulate('Hello, World!')

        import matplotlib.pyplot as plt

        if debug:

            plt.plot(sig)
            plt.show()

        print("Finished modulation")

    except:
        print('  Broken modulation -- rebuild the program')
