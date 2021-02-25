# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2016 by exopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Task perform measurements the SPDevices digitizers.

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

import numbers
import numpy as np
from inspect import cleandoc

from atom.api import (Bool, Str, Enum, set_default)

from exopy.tasks.api import InstrumentTask, validators

VAL_REAL = validators.Feval(types=numbers.Real)

VAL_INT = validators.Feval(types=numbers.Integral)

import psutil #used in the RAM estimation


class DemodAlazarTask(InstrumentTask):
    """ Get the raw or averaged quadratures of the signal.
        Can also get raw or averaged traces of the signal.
        Custom shape for demodulation can be used.
    """
    freq = Str('50').tag(pref=True)

    freqB = Str('50').tag(pref=True)

    timeaftertrig = Str('0').tag(pref=True)

    timeaftertrigB = Str('0').tag(pref=True)

    timestep = Str('0').tag(pref=True)

    timestepB = Str('0').tag(pref=True)

    tracetimeaftertrig = Str('0').tag(pref=True)

    tracetimeaftertrigB = Str('0').tag(pref=True)

    duration = Str('1000').tag(pref=True)

    durationB = Str('0').tag(pref=True)

    traceduration = Str('0').tag(pref=True)

    tracedurationB = Str('0').tag(pref=True)

    tracesbuffer = Str('20').tag(pref=True, feval=VAL_INT)

    tracesnumber = Str('1000').tag(pref=True, feval=VAL_INT)

    average = Bool(True).tag(pref=True)

    Npoints = Str('0').tag(pref=True,feval=VAL_INT)

    IQtracemode = Bool(False).tag(pref=True)

    trigrange = Enum('2.5V','5V').tag(pref=True)

    triglevel = Str('0.3').tag(pref=True, feval=VAL_REAL)

    demodFormFile = Str('[]').tag(pref=True)
    
    powerBoolA  =Bool(False).tag(pref=True) 

    powerBoolB  =Bool(False).tag(pref=True) 

    aux_trig = Bool(False).tag(pref=True)


    database_entries = set_default({'Demod': {}, 'Trace': {}, 'Power': {}})

    def format_string(self, string, factor, n):
        s = self.format_and_eval_string(string)
        if isinstance(s, list) or isinstance(s, tuple) or isinstance(s, np.ndarray):
            return [elem*factor for elem in s]
        else:
            return [s*factor]*n

    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super(DemodAlazarTask, self).check(*args,
                                                             **kwargs)
        if (self.format_and_eval_string(self.tracesnumber) %
                self.format_and_eval_string(self.tracesbuffer) != 0 ):
            test = False
            traceback[self.path + '/' + self.name + '-get_demod'] = \
                cleandoc('''The number of traces must be an integer multiple of the number of traces per buffer.''')

        if not (self.format_and_eval_string(self.tracesnumber) >= 1000):
            test = False
            traceback[self.path + '/' + self.name + '-get_demod'] = \
                cleandoc('''At least 1000 traces must be recorded. Please make real measurements and not noisy s***.''')

        time = self.format_string(self.timeaftertrig, 10**-9, 1)
        duration = self.format_string(self.duration, 10**-9, 1)
        timeB = self.format_string(self.timeaftertrigB, 10**-9, 1)
        durationB = self.format_string(self.durationB, 10**-9, 1)
        tracetime = self.format_string(self.tracetimeaftertrig, 10**-9, 1)
        traceduration = self.format_string(self.traceduration, 10**-9, 1)
        tracetimeB = self.format_string(self.tracetimeaftertrigB, 10**-9, 1)
        tracedurationB = self.format_string(self.tracedurationB, 10**-9, 1)

        for t, d in ((time,duration), (timeB,durationB), (tracetime,traceduration), (tracetimeB,tracedurationB)):
            if len(t) != len(d):
                test = False
                traceback[self.path + '/' + self.name + '-get_demod'] = \
                    cleandoc('''An equal number of "Start time after trig" and "Duration" should be given.''')
            else :
                for tt, dd in zip(t, d):
                    if not (tt >= 0 and dd >= 0) :
                           test = False
                           traceback[self.path + '/' + self.name + '-get_demod'] = \
                               cleandoc('''Both "Start time after trig" and "Duration" must be >= 0.''')

        if ((0 in duration) and (0 in durationB) and (0 in traceduration) and (0 in tracedurationB)):
            test = False
            traceback[self.path + '/' + self.name + '-get_demod'] = \
                           cleandoc('''All measurements are disabled.''')

        timestep = self.format_string(self.timestep, 10**-9, len(time))
        timestepB = self.format_string(self.timestepB, 10**-9, len(timeB))
        freq = self.format_string(self.freq, 10**6, len(time))
        freqB = self.format_string(self.freqB, 10**6, len(timeB))
        samplesPerSec = 500000000.0

        if 0 in duration:
            duration = []
            timestep = []
            freq = []
        if 0 in durationB:
            durationB = []
            timestepB = []
            freqB = []

        for d, ts in zip(duration+durationB, timestep+timestepB):
            if ts and np.mod(int(samplesPerSec*d), int(samplesPerSec*ts)):
                test = False
                traceback[self.path + '/' + self.name + '-get_demod'] = \
                   cleandoc('''The number of samples in "IQ time step" must divide the number of samples in "Duration".''')

        for f, ts in zip(freq+freqB, timestep+timestepB):
            if ts and np.mod(f*int(samplesPerSec*ts), samplesPerSec):
                test = False
                traceback[self.path + '/' + self.name + '-get_demod'] = \
                   cleandoc('''The "IQ time step" does not cover an integer number of demodulation periods.''')

        demodFormFile = self.format_and_eval_string(self.demodFormFile)

        if demodFormFile != []:
            duration=duration+durationB
            for d in duration:
                if len(demodFormFile[0]) > samplesPerSec*d:
                    test = False
                    traceback[self.path + '/' + self.name + '-get_demod'] = \
                       cleandoc('''Acquisition's duration must be larger than demodulation fonction's duration''')
        
        #RAM estimation and test
        
        #Prepare the parameters
        recordsPerCapture = self.format_and_eval_string(self.tracesnumber)
        timeA = self.format_string(self.timeaftertrig, 10**-9, 1)
        durationA = self.format_string(self.duration, 10**-9, 1)
        timeB = self.format_string(self.timeaftertrigB, 10**-9, 1)
        durationB = self.format_string(self.durationB, 10**-9, 1)
        tracetimeA = self.format_string(self.tracetimeaftertrig, 10**-9, 1)
        tracedurationA = self.format_string(self.traceduration, 10**-9, 1)
        tracetimeB = self.format_string(self.tracetimeaftertrigB, 10**-9, 1)
        tracedurationB = self.format_string(self.tracedurationB, 10**-9, 1)
        demodFormFile = self.format_and_eval_string(self.demodFormFile)

        NdemodA = len(durationA)
        if 0 in durationA:
            NdemodA = 0
            timeA = []
            durationA = []
        NdemodB = len(durationB)
        if 0 in durationB:
            NdemodB = 0
            timeB = []
            durationB = []
        NtraceA = len(tracedurationA)
        if 0 in tracedurationA:
            NtraceA = 0
            tracetimeA = []
            tracedurationA = []
        NtraceB = len(tracedurationB)
        if 0 in tracedurationB:
            NtraceB = 0
            tracetimeB = []
            tracedurationB = []
        if len(demodFormFile)== 0:
            demodCosinus = 1;
        else:
            demodCosinus = 0;

        startaftertrig = timeA + timeB + tracetimeA + tracetimeB
        duration = durationA + durationB + tracedurationA + tracedurationB
        timestepA = self.format_string(self.timestep, 10**-9, NdemodA)
        timestepB = self.format_string(self.timestepB, 10**-9, NdemodB)
        timestep = timestepA + timestepB
        freqA = self.format_string(self.freq, 10**6, NdemodA)
        freqB = self.format_string(self.freqB, 10**6, NdemodB)
        freq = freqA + freqB      
        samplesPerSec = 500000000.0         
        
        #Initialize the RAM quantities
        size_data = 0
        RAM_DEMOD = 0 #RAM coming from demodulations
        RAM_TRACE = 0 #RAM coming from traces
        RAM_list = [] #storage of the RAM estimations from each demodulation/trace
        
        #Quantities of reference 
        RAM_USAGE_1_REF = 15.265243530273438 #15.265243530273438 MB = deep_getsizeof(data,set())/(1024*1024)
                                             #where data is generated with NdemodA=1,duration=200e-9,recordsPerCapture=20000 (these are the parameters relevant to the size of data) and timestep=20ns
                                             #NOTE : the deep_getsizeof function can be found at github.com/the-gigi/deep/blob/master/deeper.py, line 80 (thanks to the author by the way)
        RAM_USAGE_2_REF = 0.1529388427734375 #0.1530609130859375 MB = deepgetsizeof(np.empty((20000,1)),set())/(1024*1024)
        
        #Calculate the RAM needed
        samplesPerDemod_RAM = []
        
        #Calculate RAM_DEMOD (in MB) coming from demodulations
        for i in range(NdemodA + NdemodB):
            samplesPerDemod_RAM.append(int(samplesPerSec * duration[i]) )

            if timestep[i] or not demodCosinus:
                RAM_DEMOD += RAM_USAGE_1_REF*(duration[i]/200e-9)*(recordsPerCapture/20000)
                RAM_list.append(RAM_USAGE_1_REF*(duration[i]/200e-9)*(recordsPerCapture/20000))                
                
            else:
                # Check wheter it is possible to cut each record in blocks of size equal
                # to an integer number of periods
                periodsPerBlock = 1
                while (periodsPerBlock * samplesPerSec < freq[i] * samplesPerDemod_RAM[i]
                        and periodsPerBlock * samplesPerSec % freq[i]):
                    periodsPerBlock += 1
                RAM_DEMOD += RAM_USAGE_2_REF*(recordsPerCapture/20000)*int(np.minimum(periodsPerBlock * samplesPerSec / freq[i],samplesPerDemod_RAM[i]))         
                RAM_list.append(RAM_DEMOD)
                #We make the assumption that the size of np.empty((a,b)) becomes linearly dependent on a and b at sufficiently high vallues
                #of a and b. Here, recordsPerCapture will not be under 1000, which guaranties this assumption
                
        #Calculate RAM_TRACE (in MB) coming from traces
        for i in (np.arange(NtraceA + NtraceB) + NdemodA + NdemodB):
            RAM_TRACE += RAM_USAGE_1_REF*(duration[i]/200e-9)*(recordsPerCapture/20000)
            RAM_list.append(RAM_USAGE_1_REF*(duration[i]/200e-9)*(recordsPerCapture/20000))
        
        size_data = RAM_DEMOD + RAM_TRACE #RAM used by the data array in MB
        
        #Determine the heights of RAM peaks linked to the wanted demodulations/traces 
        RAM_peaks = []
        for k in range(len(RAM_list)):
            RAM_peaks.append(RAM_list[k]*3+(size_data-RAM_list[k])) 
            #The demodulation/trace being calculated takes 3 times the RAM needed by its
            #data and we still have to take into account the RAM taken by the other demodulation/traces that have been or will be processed.
        estimation = max(RAM_peaks) #The higher peak is the one that matters to determine if the calculation is possible or not
        estimation = (estimation/1024) #Conversion into GB
        
        #In the case with no timestep, we have to take into account the RAM used by the buffers
        RAM_BUFFERS_REF = 1.0 #RAM (in GB) used by buffers with following parameters : recordsPerCapture = 1000000 ; duration = 1000ns
                              #(it has been measured)
        total_duration_max = np.max(np.array(startaftertrig) + np.array(duration))
        RAM_BUFFERS = RAM_BUFFERS_REF*(recordsPerCapture/1000000)*(total_duration_max/1000e-9)
        
        
        if len(timestep) != 0 and not timestep[0]: 
            estimation = max(estimation,RAM_BUFFERS) #measurements showed that the higher quantity overshadows the other and is the one 
                                                     #that matter for the RAM used in the end
        
        #Retrieval of quantities of available RAM : physical and total (physical+virtual) and conversion into GB
        RAM_physique_dispo = psutil.virtual_memory()[1]/(1024**3) 
        RAM_totale_dispo = psutil.swap_memory()[2]/(1024**3) 
        
        if (estimation+2) > RAM_physique_dispo :
            test = False
            traceback[self.path + '/' + self.name + '-get_demod'] = \
                cleandoc('''Available RAM may be insufficient. RAM needed for this calculation = '''+str(round(estimation,3))+''' (+1/-2) GB VS Available physical RAM = '''+str(round(RAM_physique_dispo,3))+''' GB and Total available RAM (physical and virtual) = '''+str(round(RAM_totale_dispo,3))+''' GB.''')
        
        return test, traceback

    def perform(self):
        """
        """
        if not self.driver:
            self.start_driver()

        if self.trigrange == '5V':
            trigrange = 5
        else:
            trigrange = 2.5

        triglevel = self.format_and_eval_string(self.triglevel)

        self.driver.configure_board(trigrange,triglevel)
        recordsPerCapture = self.format_and_eval_string(self.tracesnumber)
        recordsPerBuffer = int(self.format_and_eval_string(self.tracesbuffer))

        Npoints = self.format_and_eval_string(self.Npoints)

        timeA = self.format_string(self.timeaftertrig, 10**-9, 1)
        durationA = self.format_string(self.duration, 10**-9, 1)
        timeB = self.format_string(self.timeaftertrigB, 10**-9, 1)
        durationB = self.format_string(self.durationB, 10**-9, 1)
        tracetimeA = self.format_string(self.tracetimeaftertrig, 10**-9, 1)
        tracedurationA = self.format_string(self.traceduration, 10**-9, 1)
        tracetimeB = self.format_string(self.tracetimeaftertrigB, 10**-9, 1)
        tracedurationB = self.format_string(self.tracedurationB, 10**-9, 1)
        demodFormFile = self.format_and_eval_string(self.demodFormFile)


        NdemodA = len(durationA)
        if 0 in durationA:
            NdemodA = 0
            timeA = []
            durationA = []
        NdemodB = len(durationB)
        if 0 in durationB:
            NdemodB = 0
            timeB = []
            durationB = []
        NtraceA = len(tracedurationA)
        if 0 in tracedurationA:
            NtraceA = 0
            tracetimeA = []
            tracedurationA = []
        NtraceB = len(tracedurationB)
        if 0 in tracedurationB:
            NtraceB = 0
            tracetimeB = []
            tracedurationB = []
        if len(demodFormFile)== 0:
            demodCosinus = 1;
        else:
            demodCosinus = 0;


        startaftertrig = timeA + timeB + tracetimeA + tracetimeB
        duration = durationA + durationB + tracedurationA + tracedurationB

        timestepA = self.format_string(self.timestep, 10**-9, NdemodA)
        timestepB = self.format_string(self.timestepB, 10**-9, NdemodB)
        timestep = timestepA + timestepB
        freqA = self.format_string(self.freq, 10**6, NdemodA)
        freqB = self.format_string(self.freqB, 10**6, NdemodB)
        freq = freqA + freqB
        
        power = [self.powerBoolA,self.powerBoolB] 

        answerDemod, answerTrace, answerPower = self.driver.get_demod(startaftertrig, duration,
                                       recordsPerCapture, recordsPerBuffer,
                                       timestep, freq, self.average,
                                       NdemodA, NdemodB, NtraceA, NtraceB,
                                       Npoints,demodFormFile,demodCosinus,
                                       self.aux_trig, power)

        self.write_in_database('Demod', answerDemod)
        self.write_in_database('Trace', answerTrace)
        self.write_in_database('Power', answerPower)


class VNAAlazarTask(InstrumentTask):
    """ Allows to used an Alazar card as a VNA.
    """
    freq = Str('[]').tag(pref=True)

    freqB = Str('[]').tag(pref=True)

    timeaftertrig = Str('0').tag(pref=True)

    timeaftertrigB = Str('0').tag(pref=True)

    duration = Str('1000').tag(pref=True)

    durationB = Str('0').tag(pref=True)

    tracesbuffer = Str('20').tag(pref=True, feval=VAL_INT)

    tracesnumber = Str('1000').tag(pref=True, feval=VAL_INT)

    average = Bool(True).tag(pref=True)

    trigrange = Enum('2.5V','5V').tag(pref=True)

    triglevel = Str('0.3').tag(pref=True, feval=VAL_REAL)

    demodFormFile = Str('[]').tag(pref=True)

    aux_trig = Bool(False).tag(pref=True)

    database_entries = set_default({'VNADemod': {}})

    def format_string(self, string, factor, n):
        s = self.format_and_eval_string(string)
        if isinstance(s, list) or isinstance(s, tuple) or isinstance(s, np.ndarray):
            return [elem*factor for elem in s]
        else:
            return [s*factor]*n

    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super(VNAAlazarTask, self).check(*args,
                                                             **kwargs)

        if (self.format_and_eval_string(self.tracesnumber) %
                self.format_and_eval_string(self.tracesbuffer) != 0 ):
            test = False
            traceback[self.path + '/' + self.name + '-get_demod'] = \
                cleandoc('''The number of traces must be an integer multiple of the number of traces per buffer.''')

        if not (self.format_and_eval_string(self.tracesnumber) >= 1000):
            test = False
            traceback[self.path + '/' + self.name + '-get_demod'] = \
                cleandoc('''At least 1000 traces must be recorded. Please make real measurements and not noisy s***.''')

        time = self.format_string(self.timeaftertrig, 10**-9, 1)
        duration = self.format_string(self.duration, 10**-9, 1)
        timeB = self.format_string(self.timeaftertrigB, 10**-9, 1)
        durationB = self.format_string(self.durationB, 10**-9, 1)

        for t, d in ((time,duration), (timeB,durationB)):
            if len(t) != len(d):
                test = False
                traceback[self.path + '/' + self.name + '-get_demod'] = \
                    cleandoc('''An equal number of "Start time after trig" and "Duration" should be given.''')
            else :
                for tt, dd in zip(t, d):
                    if not (tt >= 0 and dd >= 0) :
                           test = False
                           traceback[self.path + '/' + self.name + '-get_demod'] = \
                               cleandoc('''Both "Start time after trig" and "Duration" must be >= 0.''')

        if ((0 in duration) and (0 in durationB)):
            test = False
            traceback[self.path + '/' + self.name + '-get_demod'] = \
                           cleandoc('''All measurements are disabled.''')

        demodFormFile = self.format_and_eval_string(self.demodFormFile)
        samplesPerSec = 500000000.0

        if demodFormFile != []:
            duration=duration+durationB
            for d in duration:
                if len(demodFormFile[0]) > samplesPerSec*d:
                    test = False
                    traceback[self.path + '/' + self.name + '-get_demod'] = \
                       cleandoc('''Acquisition's duration must be larger than demodulation fonction's duration''')

        return test, traceback

    def perform(self):
        """
        """
        if not self.driver:
            self.start_driver()

        if self.trigrange == '5V':
            trigrange = 5
        else:
            trigrange = 2.5

        triglevel = self.format_and_eval_string(self.triglevel)

        self.driver.configure_board(trigrange,triglevel)

        recordsPerCapture = self.format_and_eval_string(self.tracesnumber)
        recordsPerBuffer = int(self.format_and_eval_string(self.tracesbuffer))

        timeA = self.format_string(self.timeaftertrig, 10**-9, 1)
        durationA = self.format_string(self.duration, 10**-9, 1)
        timeB = self.format_string(self.timeaftertrigB, 10**-9, 1)
        durationB = self.format_string(self.durationB, 10**-9, 1)

        demodFormFile = self.format_and_eval_string(self.demodFormFile)


        NdemodA = len(durationA)
        if 0 in durationA:
            NdemodA = 0
            timeA = []
            durationA = []
        NdemodB = len(durationB)
        if 0 in durationB:
            NdemodB = 0
            timeB = []
            durationB = []

        if len(demodFormFile)== 0:
            demodCosinus = 1;
        else:
            demodCosinus = 0;


        startaftertrig = timeA + timeB
        duration = durationA + durationB

        freqA = self.format_string(self.freq, 10**6, NdemodA)
        freqB = self.format_string(self.freqB, 10**6, NdemodB)
        freq = freqA + freqB
        freqA = self.format_string(self.freq, 10**6, 1)
        if freqA != []:
            Nfreq=len(freqA)
        else:
            Nfreq = len(self.format_string(self.freqB, 10**6, 1))

        answerDemod = self.driver.get_VNAdemod(startaftertrig, duration,
                                       recordsPerCapture, recordsPerBuffer,
                                       freq, self.average, Nfreq,
                                       NdemodA, NdemodB,
                                       demodFormFile,demodCosinus,
                                       self.aux_trig)

        self.write_in_database('VNADemod', answerDemod)
        
class FFTAlazarTask(InstrumentTask):
    """ Get the raw or averaged quadratures of the signal.
        Can also get raw or averaged traces of the signal.
        Custom shape for demodulation can be used.
    """

    tracetimeaftertrig = Str('0').tag(pref=True, feval=VAL_REAL)

    tracetimeaftertrigB = Str('0').tag(pref=True, feval=VAL_REAL)

    traceduration = Str('0').tag(pref=True)

    tracedurationB = Str('0').tag(pref=True)

    tracesbuffer = Str('20').tag(pref=True, feval=VAL_INT)

    tracesnumber = Str('1000').tag(pref=True, feval=VAL_INT)

    average = Bool(True).tag(pref=True)
    
    Npoints = Str('0').tag(pref=True,feval=VAL_INT)

    trigrange = Enum('2.5V','5V').tag(pref=True)

    triglevel = Str('0.3').tag(pref=True, feval=VAL_REAL)
    
    powerPhaseA  =Bool(False).tag(pref=True)
    
    powerPhaseB = Bool(False).tag(pref=True)

    database_entries = set_default({'FFT': {}, 'freq': {},'power': {},'phase': {}})

    def format_string(self, string, factor, n):
        s = self.format_and_eval_string(string)
        if isinstance(s, list) or isinstance(s, tuple) or isinstance(s, np.ndarray):
            return [elem*factor for elem in s]
        else:
            return [s*factor]*n   
    
    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super(FFTAlazarTask, self).check(*args,
                                                             **kwargs)

        if (self.format_and_eval_string(self.tracesnumber) %
                self.format_and_eval_string(self.tracesbuffer) != 0 ):
            test = False
            traceback[self.path + '/' + self.name + '-get_demod'] = \
                cleandoc('''The number of traces must be an integer multiple of the number of traces per buffer.''')

        if not (self.format_and_eval_string(self.tracesnumber) >= 1000):
            test = False
            traceback[self.path + '/' + self.name + '-get_demod'] = \
                cleandoc('''At least 1000 traces must be recorded. Please make real measurements and not noisy s***.''')

        tracetime = self.format_string(self.tracetimeaftertrig, 10**-9, 1)
        traceduration = self.format_string(self.traceduration, 10**-9, 1)
        tracetimeB = self.format_string(self.tracetimeaftertrigB, 10**-9, 1)
        tracedurationB = self.format_string(self.tracedurationB, 10**-9, 1)

        for t, d in ((tracetime,traceduration), (tracetimeB,tracedurationB)):
            if len(t) != len(d):
                test = False
                traceback[self.path + '/' + self.name + '-get_demod'] = \
                    cleandoc('''An equal number of "Start time after trig" and "Duration" should be given.''')
            else :
                for tt, dd in zip(t, d):
                    if not (tt >= 0 and dd >= 0) :
                           test = False
                           traceback[self.path + '/' + self.name + '-get_demod'] = \
                               cleandoc('''Both "Start time after trig" and "Duration" must be >= 0.''')

        if ((0 in traceduration) and (0 in tracedurationB)):
            test = False
            traceback[self.path + '/' + self.name + '-get_demod'] = \
                           cleandoc('''All measurements are disabled.''')
        
        return test, traceback

    def perform(self):
        """
        """
        if not self.driver:
            self.start_driver()

        if self.trigrange == '5V':
            trigrange = 5
        else:
            trigrange = 2.5

        triglevel = self.format_and_eval_string(self.triglevel)

        self.driver.configure_board(trigrange,triglevel)

        recordsPerCapture = self.format_and_eval_string(self.tracesnumber)
        recordsPerBuffer = int(self.format_and_eval_string(self.tracesbuffer))
		
        Npoints = self.format_and_eval_string(self.Npoints)

        tracetimeA = self.format_string(self.tracetimeaftertrig, 10**-9, 1)
        tracedurationA = self.format_string(self.traceduration, 10**-9, 1)
        tracetimeB = self.format_string(self.tracetimeaftertrigB, 10**-9, 1)
        tracedurationB = self.format_string(self.tracedurationB, 10**-9, 1)        
        
        
        NtraceA = len(tracedurationA)
        if 0 in tracedurationA:
            NtraceA = 0
            tracetimeA = []
            tracedurationA = []
        NtraceB = len(tracedurationB)
        if 0 in tracedurationB:
            NtraceB = 0
            tracetimeB = []
            tracedurationB = []
        
        startaftertrig =tracetimeA + tracetimeB
        duration = tracedurationA + tracedurationB
        powerPhase = [self.powerPhaseA,self.powerPhaseB]
        answerFFT, answerFreq, answerFFTpower, answerFFTphase= self.driver.get_FFT(startaftertrig, duration,
                                                                                   recordsPerCapture, recordsPerBuffer,
                                                                                   self.average,
                                                                                   NtraceA, NtraceB,Npoints,powerPhase)
        self.write_in_database('FFT', answerFFT)
        self.write_in_database('freq', answerFreq)        
        self.write_in_database('power', answerFFTpower)
        self.write_in_database('phase', answerFFTphase)
    
