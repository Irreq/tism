import numpy as np

def get_main_frequency(data, Fs):

    chunk = len(data)

    data = data * np.hanning(len(data)) # smooth the FFT by windowing data
    fft = abs(np.fft.fft(data).real)
    fft = fft[:int(len(fft)/2)] # keep only first half
    freq = np.fft.fftfreq(chunk,1.0/Fs)
    freq = freq[:int(len(freq)/2)] # keep only first half
    freq = freq[np.where(fft==np.max(fft))[0][0]]+1

    return int(round(int(freq),0))
