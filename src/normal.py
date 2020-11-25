#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# src/normal.py

# Author : Irreq

"""
DOCUMENTATION:          Generate a synthetic normal distribution.

TODO:                   None
"""


import numpy as np

from sklearn.neighbors.kde import KernelDensity

from scipy.signal import argrelextrema



def constrained_sum_sample_pos(n, total):

    """
    Integer partitioning of random similar size.

    NOTE:     Return a randomly chosen list of n positive
              integers summing to total. Each such list is
              equally likely to occur.
    ARGUMENTS:
        - n:                int() The number of groups of integer that
                            together sums up to 'n'. Eg, 3

        - total:            int() An integer which will be partitioned of
                            random size. Eg, 17
    RETURNS:
        - partitioned:      list() A partition of size 'n' that sums up
                            to 'total'. Eg, [7, 4, 6]

    TODO:     None
    """

    dividers = sorted(np.random.choice(range(1, total), n - 1))

    partitioned = [a - b for a, b in zip(dividers + [total], [0] + dividers)]

    return partitioned

def constrained_sum_sample_nonneg(n, total):
    """Return a randomly chosen list of n nonnegative integers summing to total.
    Each such list is equally likely to occur."""

    return [x - 1 for x in constrained_sum_sample_pos(n, total + n)]

class KernelGenerator(object):

    def __init__(self, size=200, debug=False):

        self.data = None

        self.size = size

        self.debug = debug

        self.bias = {

            "normal_distribution" : [[0.5, 0.2],], # median, standard deviation

        }

    def kernel_normal_dist(self, mean, std, size=None, window=[0, 1]):

        """ Returns normal distribution """

        if size == None:
            size = self.size

        s = np.random.normal(mean, std, size)

        min_val, max_val = window[0], window[1]

        return [i for i in s if min_val <= i <= max_val]

    def kernel_density(self):

        pre_data = self.data

        start = np.array(pre_data)

        start_len = len(start)

        resolution = np.linspace(0, 1, num=10).tolist()

        pre_data = np.histogram(pre_data, bins=resolution)[0]

        pre_data = pre_data / max(pre_data)

        pre_data = np.array([int(i*100) for i in pre_data.tolist()])

        initial_length = int(len(pre_data) * 2) # 2 is an arbitary good number to use

        a = pre_data.reshape(-1, 1)

        kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(a)
        s = np.linspace(0, initial_length)
        e = kde.score_samples(s.reshape(-1, 1))

        lower_boundaries = argrelextrema(e, np.less)[0]

        minima = s[lower_boundaries]

        demodulated_index = [int((i/initial_length)*start_len) for i in minima]

        return start[np.array(demodulated_index)]


    def kernel_generator(self, localkernel):

        normal_distributions = []

        for k in range(len(localkernel)):

            distribution = self.kernel_normal_dist(localkernel[k][0], localkernel[k][1], self.size)

            normal_distributions.extend(distribution)


        normal_distributions.sort()

        self.data = normal_distributions[::int(round(len(normal_distributions)/self.size))]

        if len(self.data) > self.size:

            for i in range(len(self.data)-self.size):

                del self.data[np.random.randint(len(self.data))]


        elif len(self.data) < self.size:

            local_average = np.mean(np.array(self.data))

            values = self.kernel_density()[0]

            values = self.kernel_normal_dist(values, 0.1, size=self.size-len(self.data))

            values = [(i+local_average)/2 for i in values]

            self.data.extend(values)

            self.data.sort()



        if self.debug:

            import matplotlib.pyplot as plt

            print('elements : {}\nmean : {}\nmax : {}\nmin : {}'.format(len(self.data),np.mean(np.array(self.data)), max(self.data), min(self.data)))

            plt.hist(np.array(self.data), bins=np.linspace(0,1, num=100).tolist())
            plt.show()

        return self.data

    def kernel_error_catcher(self, kernel_seed):

        ErrorCount = 0

        while True:

            try:
                return self.kernel_generator(kernel_seed)

                break

            except Exception as e:

                ErrorCount += 1

                if ErrorCount > 100:
                    print('Error overflow')
                    print(e)
                    print("error found in kernel density estimation last row")
                    break




    def random_kernels(self, n_kernels):

        self.bias['normal_distribution'] = self.kernel_error_catcher(self.bias['normal_distribution'])

        def dub(n):

            return [0.2, 0.3], [0.8, 0.7]

        for i in range(n_kernels):

            x = np.random.choice(self.bias['normal_distribution'])

            print("normal_distribution")

            print(x)

            self.bias[i] = [dub(x)]

        return

    def start(self):

        for id in self.bias:

            if type(self.bias[id][0]).__name__ == list:
                continue

            self.bias[id] = self.kernel_error_catcher(self.bias[id])

    def addbias(self, *kernel):

        """
        Adds distributions from a dictionary


        kernel = eg, {'tag':[[0.4]]}

        returns = dict() # distribution

        """

        if len(kernel) == 0:
            return

        else:
            kernel = kernel[0]

        for id in kernel.keys():
            self.bias[id] = self.kernel_error_catcher(kernel[id])

        return {id:self.bias[id] for id in kernel.keys()}

    def getbias(self):

        return self.bias

    def setwindow(self, lower, upper, kernel_id):

        if type(kernel_id) == str:

            try:

                data = self.bias[kernel_id]

            except Exception as e:
                print(e)
                print("Window {} < x < {} could not be set, due to:".format(lower, upper))
                return

        elif type(kernel_id) == list:

            data = kernel_id

        else:
            data = kernel_id

        data = [i*(upper-lower)+lower for i in data]

        if type(kernel_id) == str:

            self.bias[kernel_id] = data

        return data

    def fastgen(self, distribution):

        resolution = self.size

        window = len(distribution) ** -1

        nd = self.getbias()["normal_distribution"]

        window_distribution = self.setwindow(0,window,nd)

        final_distribution = []

        for i, value in enumerate(distribution):

            data = [i*window+np.random.choice(window_distribution) for _ in range(int(value*resolution))]

            final_distribution.extend(data)

        return np.array(final_distribution)
