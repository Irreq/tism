# Signal Generation
# matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from .modulate import Modulation

from .encoding import str_to_bin

from .projects.normal_dist import KernelGenerator


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





    # data = [int(i) for i in str_to_bin('Hello, World!', encoding=True)]

    # print(data)

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
