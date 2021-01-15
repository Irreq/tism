from src import modulate

import numpy as np

from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt

# This is the string that will be generated
test_string = "Hello, World!"
test_string = [1, 0, 1, 0, 1, 1, 0, 1, 1]

# The path to save to
save_path = "demo/"

BITRATE = 30
FS = 44.1e3

mod = modulate.Modulation(samplingrate=FS, bitrate=BITRATE)

carrier = mod.modulate(test_string)

# Generate a test signal, a 2 Vrms sine wave whose frequency is slowly modulated around 3kHz, corrupted by white noise of exponentially decreasing magnitude sampled at 10 kHz.

def addnoise(carrier):
    noise_power = 0.0001 * FS / 2
    time = np.arange(len(carrier)) / float(FS)
    noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time/5)
    return carrier + noise


# Compute and plot the spectrogram.

f, t, Sxx = signal.spectrogram(carrier, FS)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
