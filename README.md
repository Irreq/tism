# Warning network is missing and will not work.

# Modem

Time-independent softmodem - tism

This is a Python implementation of a **Deep Learning Acoustic Modem** for my third year final degree project in Engineering Science and Technology, *(Teknikvetenskap)*. *(2020-2021)*

This program aims at reinventing the **Acoustic coupler modem** defined as [1]:

> In telecommunications, an acoustic coupler is an interface device for coupling electrical signals by acoustical means—usually into and out of a telephone. https://en.wikipedia.org/wiki/Acoustic_coupler

By utilizing modern computational power, this modem will ***[SOON]*** be able to transfer data faster than previous semi hardware/software modems by approaching the demodulation process in a *human-like* manner.

Theoretical transfer speeds does not directly justify this pure software implementation, however it does indirectly justify the development of demodulation technologies that could be implemented for increased *safety, transfer-speed* and reduced *error-rate*.

## Methods
Methods powering the modem are the following:

* **Deep Demodulation** using Tensorflow

* **Deep Segmentation** using Hidden Markov Models

* **Sound Activity Segmentation** using Support Vector Machines (SVM) utilizing an implemented version of silence removal from pyAudioAnalysis. Developed by **Theodoros Giannakopoulos** *(https://github.com/tyiannak/pyAudioAnalysis/)*
> @article{giannakopoulos2015pyaudioanalysis,
    title={pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis},
    author={Giannakopoulos, Theodoros},
    journal={PloS one},
    volume={10},
    number={12},
    year={2015},
    publisher={Public Library of Science}
  }

## Background

Transfer speed and loss reduction finds itself in constant improvement in **Digital Communications**. The medium in which data used to be transfered in, air is now more or less obsolete for data transfer in favour of less lossier, and faster wireless radio communication, *Wi-Fi*, *Li-Fi*. My proposal to the growing bandwidth problem is to utilize deep adaptive technologies such as deep learning for data transfer optimization, on the fly. Protocols for Wi-FI are based on [3]:

> IEEE 802.11 https://en.wikipedia.org/wiki/IEEE_802.11

However as a PoC (Proof of Concept), no initial human-like protocols are being set, the modem will itself define rules of communication during training and synchronization in production. Benefits are:

* ***Rules can change*** during transfer, not just change of rules to adapt for outside interference.

* ***Better suited rules*** I am not saying that other predefined rules or protocols are bad, however some are not as easely implemented in the acoustic spectrum.






## Installation

Clone the modem using **git clone**:

```
$ git clone https://github.com/Irreq/gyarbete.git
```

Install requirements using **pip**:

```
$ pip3 install requirements.txt
```

Install the program using **pip**:

```
$ pip3 install .
```




## Usage



The PoC Modem works in different ways:

```
$ python3 -m main.py [your_argument_here] [your_file_here] [extra_arguments_here]
```

Example :

```
$ python3 -m main.py Demodulate received_wave02.wav -s ~/Desktop/results.txt
```
Here, ```Demodulate``` tells the modem to process the wave file ```received_wave02.wav``` and save the results to the location ```~/Desktop/results.txt```

## Dependencies

    matplotlib, scipy, numpy, pyaudio, pydub, scikit-learn, keras, tensorflow


## Credits

Special thanks to the following people for making this possible.

* David Masse *(https://github.com/davidmasse/neural-net-1D-audio)*

* Theodoros Giannakopoulos *(https://github.com/tyiannak/pyAudioAnalysis/)*

* Harrison Kinsley *(https://pythonprogramming.net/)*

* Florent Forest *(https://github.com/FlorentF9/)*

## References

> [1] In telecommunications, an acoustic coupler is an interface device for coupling electrical signals by acoustical means—usually into and out of a telephone. https://en.wikipedia.org/wiki/Acoustic_coupler

> [3] IEEE 802.11 https://en.wikipedia.org/wiki/IEEE_802.11
