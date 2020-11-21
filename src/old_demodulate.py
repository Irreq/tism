import numpy as np

# from ..packaging.manchester import Manchester

from .encoding import Manchester

# for plut
import matplotlib.pyplot as plt


# for envelope_detector_and_translator
from scipy.signal import hilbert

# for filterfreq
from scipy.fftpack import rfft, irfft, fftfreq

class Demodulate(object):

    """ Analog to digital signal """

    def __init__(self,*args,override=False,debug=True,encoding=True,encryption=False):

        self._override = override
        self._debug = debug
        self._encoding = encoding
        self._encryption = encryption

        self.LOW_FREQ = 1500
        self.HIGH_FREQ = 2500
        self.DEVIATION = .25
        self.MAIN_FREQ = 2000

        self.FREQ_RANGE = [1000,2000,3000,4000]

        self._callsign = '1011010101101'
        self._callsign = '001001001'
        self._callsign = '000111000'


        for argument in args:
            try:
                self.__class__().function_manager(self.__class__.__name__,**{argument:None})
            except:
                pass


    # Sub functions

    def override(self):
        """ # Override existing values/rules """
        self._override = not self._override

    def debug(self):
        """ # Keep the instance debug or loud (printing) """
        self._debug = not self._debug

    def encoding(self):
        self._encoding = not self._encoding

    def encryption(self):
        self._encryption = not self._encryption

    def data(self):
        self._data = not self._data

    def function_manager(self,classname,**kwargs):
        """ # Run instances/classes """

        data = {}
        com = kwargs.items()
        for name, argument in com:

            value = [getattr(globals()[classname](), name)()]
            data = self.__class__().builder(value,data)

        return data



    # Main functions

    def plut(self,z,*srs, plotpath=None):

        timebins, freqbins,samples,binsize,Fs,freq = srs



        plt.imshow(z, origin="lower", aspect="auto", cmap="magma", interpolation="none")
        plt.colorbar()

        plt.xlabel("time (s)")
        plt.ylabel("frequency (hz)")
        plt.xlim([0, timebins-1])
        plt.ylim([0, freqbins])
        # plt.ylim([0, freqbins-200])

        xlocs = np.float32(np.linspace(0, timebins-1, 5))
        plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/Fs]) # Till sekunder

        ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
        plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs]) # Tar ut viktiga frekvenser

        # print(["%.02f" % freq[i] for i in ylocs])

        if plotpath:
            plt.savefig(plotpath, bbox_inches="tight")
        else:
            plt.show()

        plt.clf()

    def _final_data(self):


        d = Data()._load()

        f = self.finder(d,Fs=int(44100))

        return f

    def finder(self, samples,Fs=44100,binsize=int(2**10/2**3)):

        """ This one works for _final_data
        """

        s = Plot.spec_stft(samples, binsize)

        sshow, freq = Plot.logscale_spec(s, factor=1.0, sr=Fs)

        freq_new = []

        for i in range(len(freq)):
            if 500 < freq[i] < 6000:
                freq_new.append(freq[i])


        ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

        timebins, freqbins = np.shape(ims)

        z = np.transpose(ims)

        srs = timebins, freqbins,samples,binsize,Fs,freq

        # self.plut(z,*srs)



        z, big = self.mass_filter(z,SWITCH_VALUE=180)

        t = z.tolist()

        MAX_VALUE = 200.0 # The highest possible value


        dimensional_size = 0 # for how tall the spectrum is
        for i in range(len(t)): # Where the max values exist
            if MAX_VALUE in t[i]:
                dimensional_size += 1
        f = z.tolist()
        # print(f)
        # print([max(i) for i in f])
        # print(z)
        # print(len(z))

        # print(t)
        print(dimensional_size)

        first_data = [t[i] for i in range(len(t)) if MAX_VALUE in t[i]][int(dimensional_size/2)] # Where the max values exist and only to get the middle array and skip the first wrong element
        # first_data = [t[i] for i in range(len(t)) if MAX_VALUE in t[i]][0] # Where the max values exist and only to get the middle array and skip the first wrong element


        # print(len(first_data))



        some = [1 if i == MAX_VALUE else 0 for i in first_data] # pre translates high values to 1s and lows to 0s




        # plut(z)

        # self.plut([first_data])

        self.plut([first_data],*srs)

        data, make_avg = self.dumb_pattern_finder(some)

        avg = np.mean(make_avg)

        string = self.translator(data,avg)

        f = Demodulate().bin_to_string(string)

        return f

    def mass_filter(self,z,SWITCH_VALUE=195):

        """
        # Mega filter This one works for _final_data
        """


        # SWITCH_VALUE = 180.5
        # SWITCH_VALUE = 199.99 # Works only with generated files drop to above value if its a recorded file
        # SWITCH_VALUE = 200
        # SWITCH_VALUE = 195
        big = []


        for i in z:

            for k in range(len(i)):
                if i[k] >= SWITCH_VALUE:
                    i[k] = 200


                if i[k] < SWITCH_VALUE:
                    i[k] = 100

                big.append(i[k])

        return z, big

    def dumb_pattern_finder(self,data):

        f = {}

        location = 0

        old = None

        for i in data:
            if old == None:
                old = i

            if i != old:
                location += 1
                old = i

            try:
                main = f[location][1]

            except:
                main = 0
                pass

            f[location] = [i, main + 1]


        lo = [f[i][1] for i in f if f[i][0] == 1]

        return f, lo

    def translator(self,d,avg,data_index=0,threshold=7):

        print(d)
        print(avg)

        """
        Translates the incoming data into a string
        """

        OFF = '0'
        ON = '1'

        cal = []



        for i in d:

            place = d[i]

            value = place[1]

            if place[0] == data_index:

                if value < avg:

                    if value > avg / threshold: # Removes too short bursts
                        cal.append(OFF)

                if value > avg:

                    if value < avg * threshold: # Ignores too long bursts

                        if value > (2*avg)*0.85:

                            if (3*avg)*0.85 > value:
                                final = OFF*3

                            if (4*avg)*0.85 > value > (3*avg)*0.85:
                                final = OFF*4

                            if value > (4*avg)*0.85:
                                final = OFF*5

                            cal.append(final)

                        # if (3*avg)*0.85 > value > (2*avg)*0.85:
                        #     cal.append(OFF*3)
                        #
                        # elif (4*avg)*0.85 > value > (3*avg)*0.85:
                        #     cal.append(OFF*4)
                        #
                        # elif value > (4*avg)*0.85:
                        #     cal.append(OFF*5)
                        # # if value < avg * (threshold*2):
                        # #     cal.append(OFF*5)

                        else:

                            cal.append(OFF*2)

                    # elif value < avg * (threshold*2): # Ignores too long bursts
                    #     cal.append(OFF*4)
                    # # if value < avg * threshold and : # Ignores too long bursts
                    # #     cal.append(OFF*2)

            else:

                if value < avg:
                    if value > avg / threshold:
                        cal.append(ON)

                if value > avg:
                    if value < avg * threshold:
                        if value > (2*avg)*0.85:

                            if (3*avg)*0.85 > value:
                                final = ON*3

                            if (4*avg)*0.85 > value > (3*avg)*0.85:
                                final = ON*4

                            if value > (4*avg)*0.85:
                                final = ON*5

                            cal.append(final)

                        # if (3*avg)*0.85 > value > (2*avg)*0.85:
                        #     cal.append(ON*3)
                        #
                        # elif (4*avg)*0.85 > value > (3*avg)*0.85:
                        #     cal.append(ON*4)
                        #
                        # elif value > (4*avg)*0.85:
                        #     cal.append(ON*5)
                        # # if value < avg * (threshold*2):
                        # #     cal.append(ON*5)

                        else:

                            cal.append(ON*2)


        cal = "".join(cal) # Turns the data into a string

        return cal



    def envelope_detector_and_translator(self,sig):



        analytic_signal = hilbert(sig)

        amplitude_envelope = np.abs(analytic_signal)

        # amplitude_envelope



        # experimental

        # amplitude_envelope /= max(amplitude_envelope)

        # print(amplitude_envelope)

        result = [round(i,1) for i in amplitude_envelope] # Converts it to 0.9 and 1.0
        # print(result)

        # return result

        # print(result)

        # result = [i for i in result if 950 < i < 1050] # Experimental


        fs = 44100

        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) /
                                   (2.0*np.pi) * fs)

        t = np.arange(len(sig))

        print(len(t))

        signal = sig

        fig = plt.figure()
        ax0 = fig.add_subplot(211)
        ax0.plot(t, signal, label='huvud-signal')
        ax0.plot(t, amplitude_envelope, label='envelope-signalen')
        ax0.set_xlabel("time in seconds")
        ax0.legend()
        ax1 = fig.add_subplot(212)
        ax1.plot(t[1:], instantaneous_frequency)
        ax1.set_xlabel("tid-i-sekunder")
        ax1.set_ylim(0.0, 120.0)
        plt.show()

        return result

    def AM_translator(self,data):

        data, make_avg = self.dumb_pattern_finder(data)

        avg = np.mean(make_avg)

        print(data)

        string = self.translator(data,avg,data_index=1.0,threshold=6)

        return string


    def filter_rec(self,stream,target):



        # import numpy as np
        # from scipy.fftpack import rfft, irfft, fftfreq
        #
        # time   = np.linspace(0,10,2000)
        # signal = np.cos(5*np.pi*time) + np.cos(7*np.pi*time)
        #
        # W = fftfreq(signal.size, d=time[1]-time[0])
        # f_signal = rfft(signal)
        #
        # # If our original signal time was in seconds, this is now in Hz
        # cut_f_signal = f_signal.copy()
        # cut_f_signal[(W<6)] = 0
        #
        # cut_signal = irfft(cut_f_signal)
        #
        # import matplotlib.pyplot as plt
        # plt.subplot(221)
        # plt.plot(time,signal)
        # plt.subplot(222)
        # plt.plot(W,f_signal)
        # plt.xlim(0,10)
        # plt.subplot(223)
        # plt.plot(W,cut_f_signal)
        # plt.xlim(0,10)
        # plt.subplot(224)
        # plt.plot(time,cut_signal)
        # plt.show()









        print(len(stream))

        print(target)

        time = np.linspace(0,float(len(stream)/44100),44100)
        # signal = np.cos(5*np.pi*time) + np.cos(7*np.pi*time)

        signal = stream

        W = fftfreq(signal.size, d=time[1]-time[0])
        W = fftfreq(signal.size, d=1/44100)



        # W *= 1/max(W)

        print(max(W))
        print(len(W))

        print(np.mean(W))

        # print(W)
        # input()

        # print(W)

        f_signal = rfft(signal)

        # If our original signal time was in seconds, this is now in Hz
        cut_f_signal = f_signal.copy()
        # cut_f_signal[(W<target)] = 0
        # print(cut_f_signal)
        # cut_f_signal
        # cut_f_signal[(W>target*1.1)] = 0
        cut_f_signal[(W<994)] = 0
        cut_f_signal[(W>1006)] = 0
        # print(cut_f_signal)

        cut_signal = irfft(cut_f_signal)

        # print(max(cut_signal))

        # import matplotlib.pyplot as plt

        # plt.plot(np.arange(len(cut_f_signal)),cut_f_signal)
        # plt.plot(np.arange(len(cut_signal)),cut_signal)
        # plt.show()

        return cut_signal


    def start_AM(self,file_path):

        from scipy.io import wavfile

        Fs, stream = wavfile.read(file_path)

        # Only if the stream has been transferred


        # fig = plt.figure()
        # ax0 = fig.add_subplot(111)
        # ax0.plot(np.arange(len(stream)), stream, label='huvud-signal')

        from .filters.py_butter import AM_butter_filter

        # stream2 = self.filter_rec(stream,1000)
        # ax0.legend()
        # ax2 = fig.add_subplot(211)
        # ax2.plot(np.arange(len(stream2)), stream2, label='filter_rec')

        stream = AM_butter_filter(stream,1000)

        # ax2.legend()
        # ax1 = fig.add_subplot(212)
        # ax1.plot(np.arange(len(stream)), stream, label='AM-butter')
        # ax1.legend()
        #
        # plt.show()



        plt.plot(np.arange(len(stream)),stream)
        plt.show()

        result = self.envelope_detector_and_translator(stream)

        # print(result)

        avg = np.mean(result)

        print(avg)

        # avg = 0.65

        result = [1 if i > avg else 0 for i in result]
        # result = [1 if i > avg else 0 for i in result[::4000]]

        # print(result)

        # dd = result[::100]
        dd = result

        dd = np.abs(dd)

        print(dd)

        plt.plot(np.arange(len(dd)),np.array(dd))
        plt.show()

        result = self.AM_translator(result)



        # result = '01'+result

        # result = result[len(self._callsign):len(result)-len(self._callsign)]
        result = result[6:-6]

        print(result)

        # result = result[1:]
        # result = result[:-2]

        result = self.bin_to_string(result)

        return result

        test = ""

        test = "0100101011010101010011010100101101001101001101010100110100110101010011010011001101010010110010101010110101010101010010110100110010110010110011001011001101010010101100101100101010110010101101010101001010101011"

    # def begin_transmission(self):
    #     pass
    #
    #     return path









    def junk_new(self,final=False):

        def draw(samples,samplerate,binsize=int(2**10/2**3), plotpath=None, colormap="magma",final=False):

            import matplotlib.pyplot as plt

            s = Plot.spec_stft(samples, binsize)

            sshow, freq = Plot.logscale_spec(s, factor=1.0, sr=samplerate)

            freq_new = []

            for i in range(len(freq)):
                if 500 < freq[i] < 6000:
                    freq_new.append(freq[i])


            ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

            z = np.transpose(ims)

            print(z)

            def mass_filter(z): # Mega filter


                SWITCH_VALUE = 180.5
                # SWITCH_VALUE = 199.99 # Works only with generated files drop to above value if its a recorded file
                SWITCH_VALUE = 200
                big = []


                for i in z:
                    # print(len(i))
                    for k in range(len(i)):
                        if i[k] >= SWITCH_VALUE:
                            i[k] = 200
                            # big.append(i[k])

                        if i[k] < SWITCH_VALUE:
                            i[k] = 100

                        big.append(i[k])



                return z, big

            # if final:
            #     z, big = mass_filter(z)

            z, big = mass_filter(z)

            timebins, freqbins = np.shape(ims) # Freqbins = lines of freq, and timebins is perhaps the length

            # print("timebins: ", timebins)
            # print("freqbins: ", freqbins)

            plt.figure(figsize=(24, 8))

            # print(type(z).__name__)
            # print(len(z))
            # print(z.tolist())


            # summm = 0
            #
            # for i in z:
            #     for k in i:
            #         summm+=1
            #
            #
            # # print(z[250])
            #
            #
            # d = [i for i in z[0]]
            #
            # for i in range(len(z)):
            #     if max(z[i]) > 110:
            #         pass
                    # print(i)
            # print(min(d))

            # z.pop(510)

            # array = z

            # minima = []
            # for array in z: #where K is my array of arrays (all floats)
            #     minimum = min(array)
            #     minima = np.delete(array, minimum)
            #     minima.append(min(array))
            #
            # print(minima)
            #
            # a = np.delete(z, 510)
            #
            # z = a

            # c = [int(max(i)) for i in z]
            #
            # ff = []
            # for i in c:
            #     if i == 200:
            #         ff.append(1)
            #
            #     else:
            #         ff.append(0)
            #
            #
            # print(ff)
            # # print(c)
            #
            #
            # print(summm)

            # for i in range(len(z)):

            # z = np.array([z[i] for i in range(len(z)) if 31 < i < 62]) # Narrows down the array to only keep the frequencies in the span 25 to 65 of the 513
            # z = np.array([z[i] for i in range(len(z)) if 31 < i < 36 or 54 < i < 60]) # Narrows down the array to only keep the frequencies in the span 25 to 65 of the 513
            # print(len(z))
            # if final:
            #     z = np.array([z[i] for i in range(len(z))])
                # z = np.array([z[i] for i in range(len(z)) if 32 < i < 34 or 56 < i < 58])
                # z = np.array([z[i] for i in range(len(z)) if 32 < i < 100])


            # d = [k for i in z for k in i]
            #
            # # print(d.tolist())
            # # d.tolist()
            # new_list = [int(i) for i in d]
            # l = z
            #
            # g = [z[i] for i in range(len(z))]
            #
            # print(type(l).__name__)
            #
            #
            # z = np.array(g)
            #
            # print(type(z).__name__)
            #
            # print(l == z)
            #
            # print('looool')

            new_list = [int(k) for i in z for k in i]

            some = [1 if i == 200 else 0 for i in new_list]
            t = z.tolist()
            # print(t[10])
            dimensional_size = 0 # for how tall the spectrum is
            for i in range(len(t)): # Where the max values exist
                if 200.0 in t[i]:
                    # print(i)
                    dimensional_size += 1


            los = [t[i] for i in range(len(t)) if 200.0 in t[i]] # Where the max values exist

            # print(los)

            # print(len(los[2]))

            los = [los[int(dimensional_size/2)]] # to get the middle array and skip the first wrong element

            print(los)

            los = [los[0][1:]]

            some = [1 if int(i) == 200 else 0 for i in los[0]]

            print(some)

            # z = los

            # print(t)
            # print(some)
            # print(some)

            # print(g)
            # print(len(g))
            # print(z)
            # g = z
            #
            # print(type(g).__name__)
            #
            #
            # v = np.concatenate(g)
            # z = v
            # print(z)
            # print(len(z))
            # print(type(z).__name__)
            # print(np.array([1,2,3]))
            # plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
            def plut(z):

                plt.imshow(z, origin="lower", aspect="auto", cmap=colormap, interpolation="none")
                plt.colorbar()

                plt.xlabel("time (s)")
                plt.ylabel("frequency (hz)")
                plt.xlim([0, timebins-1])
                plt.ylim([0, freqbins])
                # plt.ylim([0, freqbins-200])

                xlocs = np.float32(np.linspace(0, timebins-1, 5))
                plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate]) # Till sekunder

                ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
                plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs]) # Tar ut viktiga frekvenser

                # print(["%.02f" % freq[i] for i in ylocs])

                if plotpath:
                    plt.savefig(plotpath, bbox_inches="tight")
                else:
                    plt.show()

                plt.clf()


            # plut(z)
            plut(los)

            # a = np.array([ 0, 47, 48, 49, 50, 97, 98, 99])
            import collections

            # some = los
            a = some
            # counter=collections.Counter(a)
            # print(counter)
            # a = [1,1,1,1,2,2,2,2,3,3,4,5,5]
            ff = np.unique(some, return_counts=True)
            # print(ff)

            # a = np.array(some)
            # print(np.split(a, np.cumsum( np.where(a[1:] - a[:-1] > 1) )+1))

            # def group_consecutives(vals, step=1):
            #     """Return list of consecutive lists of numbers from vals (number list)."""
            #     run = []
            #     result = [run]
            #     expect = None
            #     for v in vals:
            #         if (v == expect) or (expect is None):
            #             run.append(v)
            #         else:
            #             run = [v]
            #             result.append(run)
            #         expect = v + step
            #     return result

            def maximus(data):



                # data = data[:int(len(data)/6)]
                # print(data)
                f = {}

                location = 0

                old = None

                for i in data:
                    if old == None:
                        old = i

                    if i != old:
                        location += 1
                        old = i

                    try:
                        main = f[location][1]

                    except:
                        main = 0
                        pass

                    f[location] = [i, main + 1]


                # lo = []
                # for i in f:
                #
                #     if f[i][0] == 1:
                #         lo.append(f[i][1])


                lo = [f[i][1] for i in f if f[i][0] == 1]

                return f, lo

            d, lo = maximus(some)

            print(d,lo,some)

            # d.pop(0)
            # d.pop(max(d.keys()))
            # print(d)
            # print(some)

            # print(lo)

            avg = np.mean(lo)


            cal = []

            a = '0'
            b = '1'


            for i in d:

                ss = d[i][1]

                if d[i][0] == 0:

                    if ss < avg:
                        cal.append(a)

                    if ss > avg:
                        cal.append(a*2)

                else:

                    if ss < avg:
                        cal.append(b)

                    if ss > avg:
                        cal.append(b*2)


            # print(cal)

            cal = "".join(cal)
            # print(cal)

            # cal.replace('0','n',cal.count('0'))
            # cal.replace('1','e',cal.count('1'))
            # # cal.replace('n','1')
            # # cal.replace('e','0')
            #
            # print(cal)

            # s = Modulate().str_to_bin('Hello, World!')

            # print(s)

            f = Demodulate().bin_to_string(cal)
            # print(f)



            # def raw_method(data):
            #
            #     previous = None # For now, change this so it later can work with larger data sets
            #     for element in data:
            #
            #         if previous == None:
            #             previous = element

            return f

        d = Data()._load()

        f = draw(d,44100,final=final)

        return f



    def _from_npy(self):


        def fat_filter(samples,samplerate,binsize=2**10, plotpath=None, colormap="magma"):

            s = Plot.spec_stft(samples, binsize)
            # print(s)
            sshow, freq = Plot.logscale_spec(s, factor=1.0, sr=samplerate)

            # freq_new = []
            #
            # for i in range(len(freq)):
            #     if 500 < freq[i] < 6000:
            #         freq_new.append(freq[i])

            ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

            z = np.transpose(ims)

            def mass_filter(z): # Mega filter

                SWITCH_VALUE = 180.5

                big = []

                # print(z)

                for i in z:
                    # print(len(i))
                    for k in range(len(i)):
                        if i[k] > SWITCH_VALUE:
                            i[k] = 200

                        if i[k] <= SWITCH_VALUE:
                            i[k] = 100

                        big.append(i[k])

                return z, big


            z, big = mass_filter(z)

            print(big)
            print(len(big))


            # timebins, freqbins = np.shape(ims)

            freqbins = 40 # My value set below  WARNING This gives a false representation of the frequency forget this

            # z = np.array([z[i] for i in range(len(z)) if 32 < i < 34 or 56 < i < 58]) # slims it down to my desired frequencies
            # z = np.array([z[i] for i in range(len(z)) if 40 < i < 45 or 50 < i < 55])
            z = np.array([z[i] for i in range(len(z)) if i > 30])
            g = 1010110010101010
            g = 110101011011010

            print(z.tolist())

            d = z[0]

            d = []

            for i in range(len(z)):
                for k in z[i]:
                    d.append(k)

            # d.tolist()

            print(d)
            new_list = [int(i) for i in d]


            some = []

            for i in new_list:
                if i == 200:
                    some.append(1)

                else:
                    some.append(0)
            # print(some)


            def maximus(data):
                f = {}

                location = 0

                old = None

                for i in data:
                    if old == None:
                        old = i

                    if i != old:
                        location += 1
                        old = i

                    try:
                        main = f[location][1]

                    except:
                        main = 0
                        pass

                    f[location] = [i, main + 1]


                lo = []
                for i in f:

                    if f[i][0] == 1:
                        lo.append(f[i][1])




                return f, lo

            d, lo = maximus(some)
            # print(d)
            # print(some)

            # print(lo)

            avg = np.mean(lo)

            # print(avg)


            cal = []

            b = '0'
            a = '1'


            for i in d:

                c = d[i][0]

                # print(c)

                if d[i][0] == 1:

                    if d[i][1] < avg:
                        cal.append(a)

                    if d[i][1] >= avg:
                        cal.append(a*2)

                if d[i][0] == 0:

                    if d[i][1] < avg:
                        cal.append(b)

                    else:
                        cal.append(b*2)


            # print(cal)

            cal = "".join(cal)
            # print(cal)

            return cal


        def junk(d,final=False):
            # # transmission(debug=True).Recieve_Data()
            #
            #
            # dd = []
            class n:

                def __init__(self):
                    pass


            # def col(*args):
            #     print(args)
            #     print(len(args))
            #
            # col(12)
            # col()

            # n.col(12,23,23,1,23,12,31,23,123123,2)
            # n.col()

            # d = Data()._load('55.npy')
            # d = Data()._load().tolist()
            d = Data()._load()


            import matplotlib.pyplot as plt


            # print(time.time()-start)

            # def draw(samples,samplerate,binsize=2**10, plotpath=None, colormap="jet"):
            def draw(samples,samplerate,binsize=int(2**10), plotpath=None, colormap="magma",final=False):






                # samplerate, samples = data


                # samplerate, samples = read_file(file)

                s = Plot.spec_stft(samples, binsize)

                sshow, freq = Plot.logscale_spec(s, factor=1.0, sr=samplerate)

                # print(freq)


                freq_new = []

                for i in range(len(freq)):
                    if 500 < freq[i] < 6000:
                        freq_new.append(freq[i])
                    #     freq.pop(i)
                    #
                    # if i < 500:
                    #     freq.pop(i)


                # freq = freq_new


                # print(sshow)

                # print(freq)

                ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel





                z = np.transpose(ims)


                def mass_filter(z): # Mega filter


                    SWITCH_VALUE = 180.5


                    big = []


                    for i in z:
                        # print(len(i))
                        for k in range(len(i)):
                            if i[k] > SWITCH_VALUE:
                                i[k] = 200
                                # big.append(i[k])

                            if i[k] <= SWITCH_VALUE:
                                i[k] = 100

                            big.append(i[k])
                            # if i[k] > SWITCH_VALUE:
                            #     i[k] = 200
                            #     # big.append(i[k])
                            #
                            # if i[k] <= SWITCH_VALUE:
                            #     i[k] = 100
                                # big.append(i[k])

                            # if k < 600:
                            #     i[k] = 100
                            #
                            # else:






                    return z, big

                if final:
                    z, big = mass_filter(z)
                # z = mass_filter(z)

                print(z)
                # print(len(big))
                # print(big.count(200.0))
                #
                # print(big.count(200.0)/len(big))


                timebins, freqbins = np.shape(ims)


                if final:

                    freqbins = 40 # My value set below  WARNING This gives a false representation of the frequency forget this

                print("timebins: ", timebins)
                print("freqbins: ", freqbins)

                # print(np.transpose(ims))

                plt.figure(figsize=(15, 7.5))

                print(type(z).__name__)
                print(len(z))


                summm = 0

                for i in z:
                    for k in i:
                        summm+=1


                # print(z[250])


                d = [i for i in z[0]]

                for i in range(len(z)):
                    if max(z[i]) > 110:
                        print(i)
                print(min(d))

                # z.pop(510)

                # array = z

                # minima = []
                # for array in z: #where K is my array of arrays (all floats)
                #     minimum = min(array)
                #     minima = np.delete(array, minimum)
                #     minima.append(min(array))
                #
                # print(minima)
                #
                # a = np.delete(z, 510)
                #
                # z = a

                # c = [int(max(i)) for i in z]
                #
                # ff = []
                # for i in c:
                #     if i == 200:
                #         ff.append(1)
                #
                #     else:
                #         ff.append(0)
                #
                #
                # print(ff)
                # # print(c)
                #
                #
                # print(summm)

                # for i in range(len(z)):

                # z = np.array([z[i] for i in range(len(z)) if 31 < i < 62]) # Narrows down the array to only keep the frequencies in the span 25 to 65 of the 513
                # z = np.array([z[i] for i in range(len(z)) if 31 < i < 36 or 54 < i < 60]) # Narrows down the array to only keep the frequencies in the span 25 to 65 of the 513

                if final:
                    z = np.array([z[i] for i in range(len(z)) if i > 30])
                    # z = np.array([z[i] for i in range(len(z)) if 32 < i < 34 or 56 < i < 58])

                # print(len(z))
                # for i in z:
                #     print(i)


                d = z[1]

                # print(d.tolist())
                d.tolist()
                new_list = [int(i) for i in d]


                some = []

                for i in new_list:
                    if i == 200:
                        some.append(1)

                    else:
                        some.append(0)
                print(some)

                # print(g)
                # print(len(g))
                # print(z)
                # g = z
                #
                # print(type(g).__name__)
                #
                #
                # v = np.concatenate(g)
                # z = v
                # print(z)
                # print(len(z))
                # print(type(z).__name__)
                # print(np.array([1,2,3]))
                # plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")


                plt.imshow(z, origin="lower", aspect="auto", cmap=colormap, interpolation="none")
                plt.colorbar()

                plt.xlabel("time (s)")
                plt.ylabel("frequency (hz)")
                plt.xlim([0, timebins-1])
                plt.ylim([0, freqbins])
                # plt.ylim([0, freqbins-200])

                xlocs = np.float32(np.linspace(0, timebins-1, 5))
                plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate]) # Till sekunder

                ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
                plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs]) # Tar ut viktiga frekvenser

                # print(["%.02f" % freq[i] for i in ylocs])

                if plotpath:
                    plt.savefig(plotpath, bbox_inches="tight")
                else:
                    plt.show()

                plt.clf()

                # a = np.array([ 0, 47, 48, 49, 50, 97, 98, 99])
                import collections
                a = some
                # counter=collections.Counter(a)
                # print(counter)
                # a = [1,1,1,1,2,2,2,2,3,3,4,5,5]
                ff = np.unique(some, return_counts=True)
                print(ff)

                # a = np.array(some)
                # print(np.split(a, np.cumsum( np.where(a[1:] - a[:-1] > 1) )+1))

                # def group_consecutives(vals, step=1):
                #     """Return list of consecutive lists of numbers from vals (number list)."""
                #     run = []
                #     result = [run]
                #     expect = None
                #     for v in vals:
                #         if (v == expect) or (expect is None):
                #             run.append(v)
                #         else:
                #             run = [v]
                #             result.append(run)
                #         expect = v + step
                #     return result

                def maximus(data):
                    f = {}

                    location = 0

                    old = None

                    for i in data:
                        if old == None:
                            old = i

                        if i != old:
                            location += 1
                            old = i

                        try:
                            main = f[location][1]

                        except:
                            main = 0
                            pass

                        f[location] = [i, main + 1]


                    lo = []
                    for i in f:

                        if f[i][0] == 1:
                            lo.append(f[i][1])




                    return f, lo

                d, lo = maximus(some)
                # print(d)
                # print(some)

                print(lo)

                avg = np.mean(lo)


                cal = []

                b = '0'
                a = '1'


                for i in d:

                    ss = d[i][1]

                    if d[i][0] == 1:

                        if ss < avg:
                            cal.append(a)

                        else:
                            cal.append(a*2)

                    else:

                        if ss < avg:
                            cal.append(b)

                        else:
                            cal.append(b*2)


                print(cal)

                cal = "".join(cal)
                # print(cal)

                # cal.replace('0','n',cal.count('0'))
                # cal.replace('1','e',cal.count('1'))
                # # cal.replace('n','1')
                # # cal.replace('e','0')
                #
                # print(cal)

                s = Modulate().str_to_bin('Hello, World!')

                print(s)

                f = Demodulate().bin_to_string(cal)
                print(f)



                # def raw_method(data):
                #
                #     previous = None # For now, change this so it later can work with larger data sets
                #     for element in data:
                #
                #         if previous == None:
                #             previous = element


            draw(d,44100,final=final)


        # transmission(debug=True).Recieve_Data()

        # d = Data(automatic=True)._load('56.npy') #55.npy works kinda 56.npy with s[2:] works perfectly
        d = Data(automatic=True)._load() #55.npy works kinda 56.npy with s[2:] works perfectly
        # d = Data(automatic=True)._load().tolist()
        # d = Data()._load().tolist()
        print(d)
        # s = fat_filter(d,44100)

        # print(s)

        # s = s[1:]
        s = 0
        junk(s,final=True)

        # s = Modulate().str_to_bin('Hello, World!')
        print()
        print('ä')

        f = Demodulate().bin_to_string(s)

        print('crap')

        print(f)

        return f

    def checker(self, data):

        for n in range(2):

            avg = np.mean(data)
            fin = {0:[0,0,0],1:[1,1,1]}

            loc = 0
            loc_1 = 1

            HIGH_FREQ = self.HIGH_FREQ
            LOW_FREQ = self.LOW_FREQ
            DEVIATION = self.DEVIATION

            MAIN_FREQ = self.MAIN_FREQ

            storage = [HIGH_FREQ]

            rate = []


            for i in range(len(data)):
                if HIGH_FREQ * (1-DEVIATION) < data[i] < HIGH_FREQ * (1+DEVIATION):
                    if storage[-1] == LOW_FREQ:
                        real = fin[loc_1-1][1] - i

                        if n == 0:
                            fin[loc_1] = HIGH_FREQ,i,abs(real)
                        else:
                            if abs(real) > x_a_mean:
                                set = 2

                            else:
                                set = 1
                            fin[loc_1] = HIGH_FREQ,i, set

                        loc_1 += 1

                        pass
                    rate.append(HIGH_FREQ)
                    storage.append(HIGH_FREQ)
                    loc += 1



                if LOW_FREQ * (1-DEVIATION) < data[i] < LOW_FREQ * (1+DEVIATION):
                    if storage[-1] == HIGH_FREQ:

                        real = fin[loc_1-1][1] - i

                        if n == 0:
                            fin[loc_1] = LOW_FREQ,i,abs(real)
                        else:

                            if abs(real) > x_k_mean:

                                set = 2

                            else:
                                set = 1
                            fin[loc_1] = LOW_FREQ,i, set
                        loc_1 += 1

                    rate.append(LOW_FREQ)
                    storage.append(LOW_FREQ)
                    loc += 1



                if len(storage) > 4:
                    storage.pop(0)


            if n == 0:
                x_k = [fin[i][2] for i in fin if fin[i][0] == LOW_FREQ]
                x_k_mean = np.mean(x_k)
                x_a = [fin[i][2] for i in fin if fin[i][0] == HIGH_FREQ]
                x_a_mean = np.mean(x_a)
                pass

        fin = {i:(fin[i][0],fin[i][2]) for i in fin if i > 0}
        return fin

    def decompiler(self, data):

        """ Try to use K nearest neighbours to identify the sending notes """


        fin = self.__class__().checker(data)
        data = []
        for i in fin:
            if fin[i][0] == self.LOW_FREQ:
                if fin[i][1] == 1:
                    data.append(0)
                else:
                    data.append(00)

            else:
                if fin[i][1] == 1:
                    data.append(1)

                else:
                    data.append(11)


        data = "".join(str(e) for e in data)

        return data

    def bin_to_string(self, data):


        """ For Manchester Encoding

        #     Usage :
        #
        #     data = '010101010101011101010101011101010101101010101110101010101011101010101'
        #
        #     encoded_data = man_encode(data)
        #
        #     decoded_data = man_decode(encoded_data)
        #
        #     print("This is raw data : {}".format(data))
        #     print("This is modified data : {}".format(encoded_data))
        #     print("This is decoded data : {}".format(decoded_data))
        #
        #     if data == decoded_data:
        #         print("They are identical")
        #
        #     else:
        #         print("they are not identical")


        d = Demodulate(encryption=ENC).bin_to_string(d)

        print('Decoded + decryption : {}'.format(d))

        """



        if self._encoding:

            # Tries to use encoding methods,  Manchester code is the preferred initial code

            data = Manchester(differential=True).decode(data)

            if not self._debug:
                print("data is now decoded")

        string = ''.join(chr(int(data[i*8:i*8+8],2)) for i in range(len(data)//8))

        return string

    def fsk_read(self, filename):

        try:
            # If the program finds an compressed version of the file, it will try to use that file to save resources

            try_file = filename[:-4]+"-down.wav"

            # Fs is the samplingrate for the file
            # Samples are the data from the file

            Fs, samples = read(try_file)

        except:


            # Fs is the samplingrate for the file
            # Samples are the data from the file

            Fs, samples = read(filename)



        # Not entirely sure what this function does other than it makes a difference in the data

        y_diff = np.diff(samples,1)


        # Asserts the frequency bit deviation

        Fbit = 50


        # Asserts the first lowpass filter to compress the data for the first time

        y_env, h, y_filtered, N_FFT, f, w, y_f = self.__class__().lowpass(y_diff, Fbit, Fs)

        #calculate the mean of the signal
        mean = np.mean(y_filtered)

        sampled_signal = y_filtered[int(Fs/Fbit/2):int(len(y_filtered)):int(Fs/Fbit)]

        # If the mean of the bit period is higher than the mean, the data is a 0
        rx_data = [0 if bit > mean else 1 for bit in sampled_signal]



        Plot().test_plot(rx_data)

        # Turns the data sequence (list) into a string

        str_data = "".join(str(i) for i in rx_data)


        if not self._debug:

            # In debug mode, the program will display data

            print(y_diff,Fbit,Fs)


            # Program will print mean if in debug mode
            print(mean)

            print(y_filtered)
            print(len(y_filtered))
            print(max(y_filtered))
            print(min(y_filtered))
            print(int(Fs/Fbit/2))
            print(int(len(y_filtered)))
            print(int(Fs/Fbit))
            print(sampled_signal)

            print(str_data)
            print(len(str_data))


        # Binary to string conversion with encoding

        result = self.__class__(encryption=self._encryption).bin_to_string(str_data)

        if self._encryption != False: # Encryption

            result = enc.encryption_caller(result,ENC='d',version=self._encryption)

        return result

    def lowpass(self,*args):

        """
        Envelope detector + low-pass filter
        """

        y_diff, Fbit, Fs = args

        sig = y_diff

        sos = signal.butter(4, 1000, 'hp', fs=Fs, output='sos')

        filtered = signal.sosfilt(sos,sig)

        y_diff = filtered

        #create an envelope detector and then low-pass filter
        y_env = np.abs(sigtool.hilbert(y_diff))

        h=signal.firwin(numtaps=100, cutoff=Fbit*2, nyq=Fs/2)

        y_filtered=signal.lfilter( h, 1.0, y_env)

        #view the data after adding noise
        N_FFT = float(len(y_filtered))
        f = np.arange(0,Fs/2,Fs/N_FFT)
        w = np.hanning(len(y_filtered))
        y_f = np.fft.fft(np.multiply(y_filtered,w))
        y_f = 10*np.log10(np.abs(y_f[0:int(N_FFT/2)]/N_FFT))

        # t = np.arange(0,float(len(y_filtered))/float(Fbit),1/float(Fs), dtype=np.float)

        if not self._debug:
            plt.subplot(3,1,1)
            plt.plot(t[0:int(Fs*N_prntbits/Fbit)],m[0:int(Fs*N_prntbits/Fbit)])
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Original VCO output vs. time')
            plt.grid(True)
            plt.subplot(3,1,2)
            plt.plot(t[0:int(Fs*N_prntbits/Fbit)],np.abs(y[0:int(Fs*N_prntbits/Fbit)]),'b')
            plt.plot(t[0:int(Fs*N_prntbits/Fbit)],y_filtered[0:int(Fs*N_prntbits/Fbit)],'g',linewidth=3.0)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (V)')
            plt.title('Filtered signal and unfiltered signal vs. time')
            plt.grid(True)
            plt.subplot(3,1,3)
            plt.plot(f[0:int((Fc+Fdev*2)*N_FFT/Fs)],y_f[0:int((Fc+Fdev*2)*N_FFT/Fs)])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude (dB)')
            plt.title('Spectrum')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return y_env, h, y_filtered, N_FFT, f, w, y_f
