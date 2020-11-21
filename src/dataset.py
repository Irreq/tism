# Signal Generation
# matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from .modulate import Modulation

from .encoding import str_to_bin


def add_white_noise(sig, k):

    if not 0 <= k <= 1:
        print(f"K must be within 1 and 0 not {k}, k will now equal to 0.1")
        k = 0.1

    n = len(sig)

    white = np.array([np.random.random()*2-1 for i in range(n)]) * k

    mixed = white + sig * (1-k)

    mixed /= max(abs(mixed))

    return mixed

def generate_dataset(n, size):

    M = Modulation(frequency=8e3, bitrate=20)



    bin = {1:0,0:1}

    dataset = []

    names = []

    for k in range(n):

        data = []



        for i in range(size):

            if i not in [0,1]:
                if data[-1] == data[-2]:
                    data.append(bin[data[-2]])
                    continue

            data.append(np.random.choice([0,1]))

        signal = M.modulate(data)

        dataset.append(signal)

        names.append(np.array(data))

    dataset = np.array(dataset)
    names = np.array(names)

    print(dataset)
    print(dataset.shape)

    print(names)
    print(names.shape)



    # data = [int(i) for i in str_to_bin('Hello, World!', encoding=True)]

    # print(data)

def test():

    generate_dataset(20, 5)
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
