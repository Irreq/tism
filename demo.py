from src import modulate, cfg
scheme = cfg.args['scheme']

import numpy as np

from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt

from scipy.io.wavfile import write


import os

# This is the string that will be generated
test_string = "Hello, World!"
test_string = [1, 0, 1, 0, 1, 1, 0, 1, 1]

# The path to save to
save_path = "demo/"

path = r"db"

FREQUENCY = 2e3
BITRATE = 30
FS = 44.1e3
NOISE_INDEX = 0.000001

number_of_training_files = 1000

mod = modulate.Modulation(frequency=FREQUENCY, samplingrate=FS, bitrate=BITRATE)

carrier = mod.modulate(test_string)

def addnoise(carrier):
    noise_power = NOISE_INDEX * FS / 2
    time = np.arange(len(carrier)) / float(FS)
    noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time/5)
    combined = carrier + noise
    combined /= max(abs(combined))
    return combined

def append_noise(carrier):
    noise = (np.random.randn(len(y))+1)*A_n
    snr = 10*np.log10(np.mean(np.square(y)) / np.mean(np.square(noise)))
    print("SNR = %fdB" % snr)
    y=np.add(y,noise)

def test_compuation(carrier_, FS=FS):

    # test_compuation(test_audio)

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

        # signal = mod.modulate(scheme[tag]) # AM modulated data

        signal = mod.fsk_modulation(scheme[tag]) # FSK modulated data

        X = np.zeros(np.random.randint(200,300)).tolist() # add random padding for synthetic dataset

        X.extend(signal) # extend the data

        X.extend(np.zeros(np.random.randint(200, 300)).tolist()) # add random padding for synthetic dataset

        signal = addnoise(np.array(X)) # add noise to the signal, only necessary for synthetic dataset

        dataset[tag].append(signal) # append to dataset

    return dataset

def write_dataset_tofile(dataset, verbose=True):

    for key in dataset.keys():
        new_path = path + "/" + key
        try:
            os.mkdir(new_path)
        except OSError:
            if verbose:
                print(f"Creation of the directory {new_path} failed. It probably already exists.")
        else:
            if verbose:
                print(f"Successfully created the directory {new_path}")

    folders = {}

    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            folders[folder] = [files for files in os.listdir(path+"/"+folder)]

    def sort_int(examp):
        pos = 1
        while examp[:pos].isdigit():
            pos += 1
        return examp[:pos-1] if pos > 1 else examp


    tags = {}
    for sub_folder in folders:
        try:
            index = sorted(folders[sub_folder], key=sort_int)[-1] # sorts files in directory in numerical order
            tags[sub_folder] = int(index.split(".")[0]) # tries to get the integer name
        except:
            tags[sub_folder] = 0

    for key in dataset:
        for audio in dataset[key]:
            num = tags[key]
            tags[key] += 1
            filename = f"db/{key}/{num}.wav"
            if verbose:
                print(f"writing to: {filename}")
            write(filename, int(FS), audio.astype(np.int16))

            # >>> import numpy as np
            # >>> from soundfile import SoundFile
            # >>> myfile = SoundFile('stereo_file.wav')
            #
            # Write 10 frames of random data to a new file:

            # >>> with SoundFile('stereo_file.wav', 'w', 44100, 2, 'PCM_24') as f:
            # >>>     f.write(np.random.randn(10, 2))



    for i in dataset:
        print(f"{i}:{len(dataset[i])}, avg length:{np.mean([len(l) for l in dataset[i]])/FS}")

if __name__ == "__main__":

    complete_dataset = generate(number_of_training_files)

    write_dataset_tofile(complete_dataset)
