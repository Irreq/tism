#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# main.py

# Author : Irreq

"""
DOCUMENTATION:          I propose a deep learning method for
                        asynchronus demodulation of AM signals.

TODO:                   Implement the CNN and fix the core of
                        modem.
"""

__author__ = "Isac Bruce"
__copyright__ = "Copyright 2020, Irreq"
__credits__ = ["Isac Bruce"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Isac Bruce"
__email__ = "irreq@protonmail.com"
__status__ = "Development"


import sys, os


import pyaudio

import numpy as np

import argparse

from concurrent.futures import ThreadPoolExecutor


from src import buffer

from src import modulate#, #demodulate, dataset




pa = pyaudio.PyAudio()

__path__ = os.path.dirname(os.path.abspath(__file__))

if __path__[-1] != '/':
    __path__ += '/'




# ============= START SEND DATA ====================

def callback(in_data, frame_count, time_info, status):

    """
    Datastream creator + underrun protector.

    NOTE:     This function is threaded and should not be
              initiated by any other than:
              send()

              Documentation gathered from:
              https://people.csail.mit.edu/hubert/pyaudio/docs/#pasampleformat

    ARGUMENTS:
        - in_data:           -

        - frame_count:       int()

        - time_info:         -

        - status:            -

    RETURNS:
        - tuple():           Must return a tuple containing frame_count
                             frames of audio data and a flag signifying
                             whether there are more frames to play/record.

    TODO:     None
    """

    data = buffer.stream_to_send[:frame_count]

    size = len(data)

    del buffer.stream_to_send[:size]

    data.extend(np.zeros(frame_count-size).tolist())

    return (np.array(data).astype(np.float32).tobytes(), pyaudio.paContinue)

def send():

    """
    Process and send data.

    NOTE:     This function is threaded and should not be
              initiated by any other than:
              run_io_tasks_in_parallel()

    TODO:     None
    """

    print('output is initiating')

    buffer.streams.append(pa.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=int(buffer.args['samplingrate']),
                                  output=True,
                                  stream_callback=callback))

    print('output stream has been initiated')

    Mod = modulate.Modulation(
                              frequency=buffer.args['frequency'],
                              samplingrate=buffer.args['samplingrate'],
                              bitrate=buffer.args['bitrate'],
                              amplitude=buffer.args['amplitude'],
                              encoding=buffer.args['encoding'],
                              )

    print('modulation has been initiated')

    while buffer.status != False:

        if buffer.status == 'send':

            if len(buffer.stream_to_modulate) != 0:

                bitstream = buffer.stream_to_modulate[0]

                del buffer.stream_to_modulate[0]

                signal = Mod.modulate(bitstream)

                buffer.stream_to_send.extend(signal.tolist())

            buffer.status = 'receive'

    print('output has been shutdown')

# ============== END SEND DATA ====================



# ============= START RECEIVE DATA ================

def receive():

    """
    Receive and process data.

    NOTE:     This function is threaded and should not be
              initiated by any other than:
              run_io_tasks_in_parallel()

    TODO:     Fix the receiving end.
    """

    print('input is initiating')

    while buffer.status != False:
        pass

        # if buffer.status != 'receive':
        #     stream.stop_stream()
        #     stream.close()

    print('input has been shutdown')

# ============== END RECEIVE DATA ================



# ============= START CORE FUNCTIONS =================

def parse_arguments():

    """
    Argument parser for initiating the modem.

    NOTE:     This function modifies a global
              dictionary containing variables
              in src/buffer.py as args.

    TODO:     Add more arguments
    """

    parser = argparse.ArgumentParser()

    # Signal
    parser.add_argument('--samplingrate', default=44.1e3, type=float, help='samplingrate')
    parser.add_argument('--encoding', default='manchester', type=str, help='encoding method')
    parser.add_argument('--bitrate', default=100.0, type=float, help='transfer bitrate')
    parser.add_argument('--frequency', default=1e3, type=float, help='signal frequency')
    parser.add_argument('--amplitude', default=1.0, type=float, help='signal amplitude')

    # Training
    parser.add_argument('--modelpath', default=__path__ + 'src/tmp/', type=str, help='encoding method')
    parser.add_argument('--batchsize', default=1e3, type=float, help='train size')
    parser.add_argument('--snr', default=1.0, type=float, help='signal to noise ratio')

    # Miscellanious
    parser.add_argument('--debug', default=False, type=bool, help='debug mode')
    parser.add_argument('--plot', default=False, type=bool, help='plot mode')

    args = parser.parse_args().__dict__

    args['path'] = __path__

    buffer.args = args

    buffer.status = True

def run_io_tasks_in_parallel(tasks):

    """
    Run functions in paralell.

    NOTE:     This function can be used for any type of functions.

    ARGUMENTS:
        - tasks:             list() representing function variables.
                             Eg, [foo, bar]

    TODO:     None
    """

    with ThreadPoolExecutor() as executor:
        running_tasks = [executor.submit(task) for task in tasks]
        for running_task in running_tasks:
            running_task.result()

def terminate():

    """
    Terminate the modem.

    NOTE:     This function is threaded and should not be
              initiated by any other than:
              run_io_tasks_in_parallel()

    TODO:     None
    """

    import time

    time.sleep(5)

    print('Modem is now terminating')

    buffer.status = False

    for stream in buffer.streams:
        if stream.is_active():
            stream.stop_stream()
            stream.close()


    pa.terminate()

    print('Modem has been terminated successfully!')
    exit()

# ============== END CORE FUNCTIONS =================





if __name__ == '__main__':

    print('This is a modem')

    parse_arguments()

    run_io_tasks_in_parallel([
                              send,
                              receive,
                              terminate,
                              ])
