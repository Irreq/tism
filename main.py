#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# main.py

"""
    Documentation

    I propose a deep learning method for
    asynchronus demodulation of AM signals.

"""


__author__ = "Isac Bruce"
__copyright__ = "Copyright 2020, Irreq"
__credits__ = ["Isac Bruce"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Isac Bruce"
__email__ = "irreq@protonmail.com"
__status__ = "Development"



import sys

try:

    # import tensorflowi as tf
    pass

except ImportError as e:

    print(f"{str(e)} could not be found, try installing it using pip.")

    quit()



from src import modulate, demodulate, dataset

def start(*args):

    # M = modulate.Modulation(bitrate=100, frequency=8e3)
    #
    # samplingrate = M.samplingrate
    #
    # sig = M.modulate('hello')
    #
    # energy = abs(sig)
    #
    # import matplotlib.pyplot as plt
    #
    # # plt.specgram(sig, Fs=44.1e3)
    # # # plt.plot(energy)
    # # plt.show()
    #
    # freq = demodulate.get_main_frequency(sig, samplingrate)
    #
    # print(freq)



    dataset.test()




if __name__ == '__main__':

    start(sys.argv[1:])
