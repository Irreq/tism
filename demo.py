from src import modulate, cfg
scheme = cfg.args['scheme']

import numpy as np

from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt

# This is the string that will be generated
test_string = "Hello, World!"
test_string = [1, 0, 1, 0, 1, 1, 0, 1, 1]

# The path to save to
save_path = "demo/"

FREQUENCY = 2e3
BITRATE = 30
FS = 44.1e3
NOICE_INDEX = 0.000001

mod = modulate.Modulation(frequency=FREQUENCY, samplingrate=FS, bitrate=BITRATE)

carrier = mod.modulate(test_string)

# Generate a test signal, a 2 Vrms sine wave whose frequency is slowly modulated around 3kHz, corrupted by white noise of exponentially decreasing magnitude sampled at 10 kHz.

def addnoise(carrier):
    noise_power = NOICE_INDEX * FS / 2
    time = np.arange(len(carrier)) / float(FS)
    noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time/5)
    combined = carrier + noise
    combined /= max(abs(combined))
    return combined

def test_compuation(carrier_, FS=FS):

    # Compute and plot the spectrogram.

    f, t, Sxx = signal.spectrogram(carrier_, FS)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def generate(n):

    dataset = {**scheme}

    for key in dataset.keys():
        dataset[key] = []

    for i in range(n):

        tag = np.random.choice(list(scheme.keys()))

        signal = mod.modulate(scheme[tag]) # modulated data

        X = np.zeros(np.random.randint(200,300)).tolist() # add random padding for synthetic dataset

        X.extend(signal) # extend the data

        X.extend(np.zeros(np.random.randint(200, 300)).tolist()) # add random padding for synthetic dataset

        signal = addnoise(np.array(X)) # add noise to the signal, only necessary for synthetic dataset
        # signal = signal.tolist().extend(np.zeros(np.random.randint(50, 100)).tolist())
        signal = np.float32(signal) # convert to correct format

        dataset[tag].append(signal) # append to dataset

    return dataset




dataset = generate(10)

for i in dataset:
    print(f"{i}:{len(dataset[i])}, avg length:{np.mean([len(l) for l in dataset[i]])/FS}")

test_audio = dataset['0'][0]
# plt.plot(test_audio)
# spectrum2-D array
#
#     Columns are the periodograms of successive segments.
# freqs1-D array
#
#     The frequencies corresponding to the rows in spectrum.
# t1-D array
#
#     The times corresponding to midpoints of segments (i.e., the columns in spectrum).
spectrum2d, freqs1d, t1, im = plt.specgram(test_audio)
print()
# plt.plot(spectrum2d, freqs1d)
#
# plt.show()

def sort_numerical():
    def sort_int(examp):
        pos = 1
        while examp[:pos].isdigit():
            pos += 1
        return examp[:pos-1] if pos > 1 else examp

    sorted(files, key=sort_int)

from scipy.io.wavfile import write

samplerate = 44100; fs = 100

t = np.linspace(0., 1., samplerate)

amplitude = np.iinfo(np.int16).max

data = amplitude * np.sin(2. * np.pi * fs * t)

tag = "0"
data = dataset[tag][-1] # the last element

tags = {i:0 for i in dataset.keys()}

print(num)

# write(".db/"+tag+"/"+num+".wav", FS, data.astype(np.float32))

for key in dataset:
    for audio in dataset[key]:
        num = tags[key]
        tags[key] += 1
        write(".db/"+key+"/"+num+".wav", FS, data.astype(np.float32))


# test_compuation(test_audio)
