#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# src/window.py

# Author : Irreq

"""
DOCUMENTATION:          Handle windowing and padding for demodulation
                        and training.

TODO:                   Optimize for speed.
"""

# =============== Start Windowing =======================

class Window(object):

    """
    Windowclass for demodulation
    """

    def __init__(self, winsize, steps=False, padding=0):

        """
        Window data for demodulation

        Note:                   This class has a built in padding function
                                which enables equal window size across all
                                segments.

        ARGUMENTS:
            - winsize           int() The size of the window. Eg, 4410

        KEYWORD ARGUMENTS:
            - steps             int() How many indexes the window will jump.
                                Eg, 512

            - padding           Anything that will be used as padding.
                                Eg, 0
        """

        self.winsize = winsize

        if not steps:
            steps = 512

        else:

            assert 0 < steps < self.winsize, "steps must be greater than zero and lower than window"

        self.steps = steps
        self.padding = padding

        self.data = None
        self.segments = []


    def _parts(self, i):

        """
        Return parts of list

        Note:                   This function is not recomended to be called by the user.

        ARGUMENTS:
            - i                 int() The position in which the data will be collected.

        RETURNS:
            - list()            list() Containing the selected data.

        """

        return self.data[int(i*self.steps):int(i*self.steps+self.winsize)]


    def _wrapper(self):

        """
        Process the segments

        Note:                   This function is not recomended to be called by the user.
                                The function works as a wrapper for the other functions.
                                See further documentation below:
        """

        # The number of times the function will loop
        # as in this case any arbituary number

        times = int(len(self.data)/self.steps)

        for index in range(times):

            segment = self._parts(index)

            # Checks if the returned segment is shorter than the window
            # padding will be added if true and break the loop

            if len(segment)%self.winsize > 0:

                self.segments.append(self.pad(segment))

                break

            self.segments.append(segment)


    def pad(self, data):

        """
        Pad the data

        Note:                   This function pads data when size is
                                insufficient. This function is not recomended
                                to call if the size of the window as been altered
                                ("winsize"), The padding method was declared in
                                the "__init__" function.
        ARGUMENTS:
            - data              list() The list that will have padding added to it

        RETURNS:
            - list()            list() That has the correct length.
        """

        # Unpacks the previous list combined with the correct number of paddings

        return [*data, *[self.padding,]*int(self.winsize-len(data))]


    def segment(self, data, clear=False):

        """
        Window data

        Note:                   This is the function to call when segmenting.

        ARGUMENTS:
            - data              numpy.ndarray() The incoming data stream.

        KEYWORD ARGUMENTS:
            - clear             Boolean() If the class should clear itself after use
                                in order to free up ram. If this class is being called
                                by anyone other than the demodulation, it is suitable to
                                flush data. Eg, "clear=True"
        RETURNS:
            - list()            list() Containing the windows to be processed.

        TODO:                   Fix some cleanup with the function for easier readabbility
                                and performance optimization.

        """

        self.data = data

        self._wrapper()

        values = self.segments

        if clear:
            self.flush()

        return values


    def flush(self, n=-1):

        """
        Clear used variable data

        Note:                   This function is not recomended to be called by the user
                                if the class is being used elsewhere.

        KEYWORD ARGUMENTS:
            - n                 int() The last index which all lists prior to "n" will
                                be deleted. If nothing is entered the function will delete
                                all data.
        """

        # Data cleanup

        self.data = None

        self.segments = self.segments[n:-1]

# ================ End Windowing =======================
