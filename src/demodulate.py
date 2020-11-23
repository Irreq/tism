#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# src/demodulate.py

# Author : Irreq

"""
DOCUMENTATION:          Deep learning demodulation of signal to
                        binary sequence.

TODO:                   Add more demodulation schemes and comment
                        the program for easier readability.
                        Implement CNN.
"""

import numpy as np

from .silence import silence_removal

def get_main_frequency(data, Fs):

    chunk = len(data)

    data = data * np.hanning(len(data)) # smooth the FFT by windowing data
    fft = abs(np.fft.fft(data).real)
    fft = fft[:int(len(fft)/2)] # keep only first half
    freq = np.fft.fftfreq(chunk,1.0/Fs)
    freq = freq[:int(len(freq)/2)] # keep only first half
    freq = freq[np.where(fft==np.max(fft))[0][0]]+1

    return int(round(int(freq),0))


segments = silence_removal(x, Fs, 0.02, 0.09, smooth_window = 0.6, weight = 0.4, plot = False)

segments = [[int(i[0]*Fs), int(i[1]*Fs)] for i in segments]

d = [0 if -0.8 < i < 0.8 else i for i in x[segments[0][0]:segments[0][1]]]
g = np.zeros(Fs).tolist()

g.extend(d)
g.extend(np.zeros(Fs).tolist())

signal = np.array(g)

signal = add_white_noise(signal, 0.01)

# plt.plot(signal)
# plt.show()

segments2 = silence_removal(signal, Fs, 0.01, 0.1, smooth_window = 0.20, weight = 0.5, plot = False)

print(segments2)

segments2 = [[int(i[0]*Fs), int(i[1]*Fs)] for i in segments2]

plut([[x, segments],[signal, segments2]])

avg = np.mean([i[1]-i[0] for i in segments2])


bits = []
size = len(segments2)
for i in range(size):

  if i == size-1:
    break

  avg += segments2[i+1][0] - segments2[i][1]

  avg /= 2

for i in range(size):

  lol = segments2[i][1]-segments2[i][0]

  if lol > avg:
    bits.extend([1,1])
  else:
    bits.extend([1])

  if i < size-1:
    if segments2[i+1][0] - segments2[i][1] > avg:
      bits.extend([0,0])
    else:
      bits.extend([0])








# avg = np.mean(bits)

print('test {}'.format(str(bits)))
print('True {}'.format(str(send_data)))
