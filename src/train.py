#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# src/train.py

# Author : Irreq

"""
DOCUMENTATION:          Train the model. Generate dataset for
                        training.

TODO:                   Fix structure and ease readability and
                        optimize for performance. Implement the
                        CNN and fix the core of modem.
"""


import numpy as np

import pandas as pd

from keras.preprocessing.sequence import pad_sequences

# import matplotlib.pyplot as plt

from .modulate import Modulation

from .encoding import str_to_bin

from .normal import KernelGenerator

from . import cfg as buffer

def add_white_noise(sig, k):

    if not 0 <= k <= 1:
        print(f"K must be within 1 and 0 not {k}, k will now equal to 0.1")
        k = 0.1

    n = len(sig)

    white = np.array([np.random.random()*2-1 for i in range(n)]) * k

    mixed = white + sig * (1-k)

    if max(abs(mixed)) > 1:

        mixed /= max(abs(mixed))

    return mixed

def generate_dataset(n, size):

    kg = KernelGenerator()

    kg.start()

    bias = kg.fastgen([0.1, 0.4, 0.3, 0.2])

    bias = kg.setwindow(0, 0.5, bias)

    M = Modulation(frequency=8e3, bitrate=20)

    bin = {1:0,0:1}

    streams = []

    names = []

    final = []

    for k in range(n):

        data = []

        for i in range(size):

            if i not in [0,1]:
                if data[-1] == data[-2]:
                    data.append(bin[data[-2]])
                    continue

            data.append(np.random.choice([0,1]))

        signal = M.modulate(data)

        signal = add_white_noise(signal, np.random.choice(bias))

        streams.append(signal)

        names.append(np.array(data))

        data.append(signal)

        final.append(np.array(data))

    return np.array(final)

    sig_data = np.array(streams)
    sig_tag = np.array(names)

    return sig_data, sig_tag

def test():

    # dataset, tags = generate_dataset(1, 5)
    #
    # print(dataset)
    # print(dataset.shape)
    #
    # print(tags)
    # print(tags.shape)

    dataset = generate_dataset(1, 5)

    print(dataset)
    print(dataset.shape)

    return



    import matplotlib.pyplot as plt

    plt.specgram(dataset[-1])
    plt.show()


    return

    M = Modulation(frequency=8e3, bitrate=20)

    signal = M.modulate([1,1,0])
    plt.subplot(221)

    plt.specgram(signal, Fs=M.samplingrate)

    plt.subplot(222)
    plt.plot(signal)

    signal = add_white_noise(signal, .2)

    print(max(signal), min(signal))

    # signal = signal + noise
    #
    # signal /= max(abs(signal))
    #
    # print(max(signal), min(signal))

    plt.subplot(223)

    plt.specgram(signal, Fs=M.samplingrate)

    plt.subplot(224)
    plt.plot(signal)
    plt.show()

def generate(n):
    # 0 = 0, 1 = 1, 00, = 2, 11 = 3

    scheme = buffer.args['scheme']
    # padding = buffer.args['padding']

    data_x = []

    data_y = []

    M = Modulation(frequency=2e3, bitrate=50)

    kg = KernelGenerator()
    kg.start()
    bias = kg.fastgen([0.1, 0.4, 0.3, 0.2])
    bias = kg.setwindow(0, 0.4, bias)

    for i in range(n):

        tag = np.random.choice(list(scheme.keys()))

        data_y.append(tag)

        signal = M.modulate(scheme[tag])

        # print(np.zeros(np.random.randint(50, 1000)).tolist())

        result = np.zeros(np.random.randint(50,100)).tolist()

        result.extend(signal)

        result.extend(np.zeros(np.random.randint(50, 100)).tolist())

        signal = np.array(result)

        # signal = add_white_noise(signal, np.random.choice(bias))

        data_x.append(np.float32(signal))

    return data_x, data_y

def genset():
    # data_x = [] # continious stream of modulated signals as either 0, 00, 1, 11

    # data_y = [] # The corresponding labels but will be modulated as 0 = 0, 1 = 1, 00, = 2, 11 = 3


    size = buffer.args['batchsize']
    data_x, data_y = generate(size)

    # preprocessing data



    data_x = pad_sequences(data_x, maxlen=int(size*1.1), dtype='float', padding='post', truncating='post', value=0.)

    data_x = data_x / np.max(data_x)

    data_x = data_x[:,:,np.newaxis]



    # Labeling
    # 0 = 0, 1 = 1, 00, = 2, 11 = 3

    data_y = pd.Series(data_y)
    data_y.value_counts()
    scheme = buffer.args['scheme']
    ff = {i:scheme[i][0] for i in scheme}
    data_y = data_y.map(ff).values
    # data_y = data_y.map({'0':0, '1':1, '00':2, '11':3}).values

    return data_x, data_y

def generate_dataset_for_folder(n):

    scheme = buffer.args['scheme']

    data_x = []

    data_y = []

    M = Modulation(frequency=2e3, bitrate=50)

    for i in range(n):

        tag = np.random.choice(list(scheme.keys()))

        data_y.append(tag)

        signal = M.modulate(scheme[tag])

        # print(np.zeros(np.random.randint(50, 1000)).tolist())

        result = np.zeros(np.random.randint(50,100)).tolist()

        result.extend(signal)

        result.extend(np.zeros(np.random.randint(50, 100)).tolist())

        signal = np.array(result)

        # signal = add_white_noise(signal, np.random.choice(bias))

        data_x.append(np.float32(signal))

    return data_x, data_y
