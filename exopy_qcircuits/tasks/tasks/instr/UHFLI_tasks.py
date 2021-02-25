# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by exopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, diStributed with this software.
# -----------------------------------------------------------------------------
"""Task perform measurements for the UFHLI zurick instrument.

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

import numbers
import numpy as np
import textwrap
import time
from inspect import cleandoc
from collections import OrderedDict
import os

from atom.api import (Bool, Str, Enum, set_default,Typed)

from exopy.tasks.api import InstrumentTask, validators
from exopy.utils.atom_util import ordered_dict_from_pref, ordered_dict_to_pref


VAL_REAL = validators.Feval(types=numbers.Real)

VAL_INT = validators.Feval(types=numbers.Integral)# -*- coding: utf-8 -*-


def eval_with_units(task,evaluee):
    value = task.format_and_eval_string(evaluee[0])
    unit = str(evaluee[1])

    unitlist = ['','none','ns','µs','s','kHz','MHz','GHz','mV','V','ns_to_clck','clock_samples'] #from views
    multlist  = [1,1,1e-9,1e-6,1,1e3,1e6,1e9,1e-3,1,0.225,1] #corresponding to above list
    multiplier = multlist[np.argwhere(np.array(unitlist)==unit)[0,0]]
    value = value * multiplier
    return value

class ScopeDemodUHFLITask(InstrumentTask):
    """ Get the raw or averaged quadratures of the signal.
        Can also get raw or averaged traces of the signal.
    """
    freq = Str('25').tag(pref=True)

    freq2 = Str('25').tag(pref=True)

    delay = Str('0').tag(pref=True)

    delay2 = Str('0').tag(pref=True)

    duration = Str('1000').tag(pref=True)

    duration2 = Str('0').tag(pref=True)

    tracesnumber = Str('1000').tag(pref=True, feval=VAL_INT)

    average = Bool(True).tag(pref=True)

    demod = Bool(True).tag(pref=True)

    trace = Bool(False).tag(pref=True)

    demod2 = Bool(True).tag(pref=True)

    trace2 = Bool(False).tag(pref=True)

    Npoints = Str('1').tag(pref=True,feval=VAL_INT)

    customDemod = Str('[]').tag(pref=True)

    samplingRate = Enum('1.8GHz','900MHz','450MHz','225MHz','113MHz',
                        '56.2MHz','28.1MHz','14MHz','7.03MHz','3.5MHz',
                        '1.75MHz','880kHz','440kHz','220kHz','110kHz',
                        '54.9kHz','27.5kHz').tag(pref=True)


    database_entries = set_default({'Demod': {}, 'Trace': {}})
    def format_multiple_string(self, string, factor, n):
        s = self.format_and_eval_string(string)
        if isinstance(s, list) or isinstance(s, tuple) or isinstance(s, np.ndarray):
            return [elem*factor for elem in s]
        else:
            return [s*factor]*n

    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super(ScopeDemodUHFLITask, self).check(*args,
                                                             **kwargs)

        if not (self.format_and_eval_string(self.tracesnumber) >= 1000):
            test = False
            traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                cleandoc('''At least 1000 traces must be recorded. Please make real measurements and not noisy s***.''')
        duration = np.array(self.format_and_eval_string(self.duration))
        duration2 = np.array(self.format_and_eval_string(self.duration2))
        if (0 in duration and 0 in duration2):
            test = False
            traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                           cleandoc('''All measurements are disabled.''')
        recordsPerCapture = self.format_and_eval_string(self.tracesnumber)
        Npoints = self.format_and_eval_string(self.Npoints)

        if int(recordsPerCapture/Npoints)!=recordsPerCapture/Npoints:
            test = False
            traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                           cleandoc('''Number of traces musst be divisible by Npoints.''')

        customDemod = self.format_and_eval_string(self.customDemod)
        samplingRate = self.samplingRate
        if samplingRate[-3]=='G':
            factor = 10**9
        elif samplingRate[-3]=='M':
            factor = 10**6
        else:
            factor = 10**3

        samplingRate = float(samplingRate[:-3])*factor
        if customDemod != []:
            if len(customDemod[0]) != samplingRate*np.sum(duration):
                test = False
                traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                   cleandoc('''Acquisition's duration must be the same as demodulation fonction's duration''')
        return test, traceback

    def perform(self):
        """
        """
        if not self.driver:
            self.start_driver()
        if not self.driver.saveConfig:
            path = self.format_string('{default_path}')
            measId = self.format_string('LabOne_settings_{meas_id}.xml')
            full_path = os.path.join(path,measId)
            self.driver.saveConfiguration(full_path)

        self.driver.set_general_setting();

        recordsPerCapture = self.format_and_eval_string(self.tracesnumber)
        Npoints = self.format_and_eval_string(self.Npoints)
        delay = self.format_and_eval_string(self.delay)*10**-9
        duration = np.array(self.format_multiple_string(self.duration,10**-9,1))
        delay2 = self.format_and_eval_string(self.delay2)*10**-9
        duration2 = np.array(self.format_multiple_string(self.duration2,10**-9,1))
        customDemod = self.format_and_eval_string(self.customDemod)
        samplingRate = self.samplingRate
        freq = self.format_and_eval_string(self.freq)*10**6
        freq2 = self.format_and_eval_string(self.freq2)*10**6

        if duration.shape == ():
            duration = np.array([duration])
        if duration2.shape == ():
            duration2 = np.array([duration2])

        if samplingRate[-3]=='G':
            factor = 10**9
        elif samplingRate[-3]=='M':
            factor = 10**6
        else:
            factor = 10**3

        samplingRate = float(samplingRate[:-3])*factor
        indexSamplingRate = int(round(np.log(1.8*10**9/samplingRate)/np.log(2)))
        #give experimental setting
        length = int(np.max([(np.sum(duration)+delay),(np.sum(duration2)+delay2)])*samplingRate)
        device = self.driver.device
        if 0 in duration:
            channel = 2
        elif 0 in duration2:
            channel=1
        else :
            channel = 3
        exp_setting = [['/%s/scopes/0/length'          % device, length+10],# number of points
                        ['/%s/scopes/0/channel'         % device, channel],# 1-> channel 0,2-> channel 1, 3-> channel 1 and 0
                        ['/%s/scopes/0/channels/%d/inputselect' % (device, 0), 0],# input 1 selected channel 1
                        ['/%s/scopes/0/channels/%d/inputselect' % (device, 1), 1],# input 2 selected channel 2
                        ['/%s/scopes/0/time'            % device, indexSamplingRate],#  sampling rate
                        ['/%s/scopes/0/trigholdoffmode' % device, 1],#trigger hold off mode
                        ['/%s/scopes/0/trigholdoffcount'% device, 0],# re-arming the scope every 1 trigger
                        ['/%s/scopes/0/trigdelay'       % device, 0],
                        ['/%s/scopes/0/trigenable'      % device, 1],
                        ['/%s/scopes/0/trigreference'   % device, 0],
                        ['/%s/scopes/0/segments/enable' % device, 0],# segment
                        ['/%s/scopes/0/segments/count'  % device, 1]]# nb segment
        self.driver.daq.set(exp_setting)
        self.driver.daq.sync()



        if len(customDemod)== 0:
            demodCosinus = 1;
        else:
            demodCosinus = 0;

        scopeModule = self.driver.daq.scopeModule()
        scopeModule.set('scopeModule/mode', 1)# averager weight by default equal to 10
        scopeModule.set('scopeModule/historylength', 1)# memory
        scopeModule.set('scopeModule/averager/weight', 1)# disable average


        # Perform a global synchronisation between the device and the data server:
        # Ensure that the settings have taken effect on the device before acquiring
        # data.
        wave_nodepath = '/{}/scopes/0/wave'.format(self.driver.device) #path for the data
        scopeModule.unsubscribe('*')
        self.driver.daq.unsubscribe('/dev2375/*')
        self.driver.daq.flush()
        scopeModule.subscribe(wave_nodepath)
        self.driver.daq.sync()
        answerDemod, answerTrace = self.driver.get_scope_demod(samplingRate, [duration,duration2], [delay, delay2], recordsPerCapture,
                                                               [freq, freq2], self.average,[self.demod, self.demod2],
                                                               [self.trace, self.trace2], Npoints,customDemod,demodCosinus,scopeModule)

        self.write_in_database('Demod', answerDemod)
        self.write_in_database('Trace', answerTrace)


class DemodUHFLITask(InstrumentTask):
    """ Get the demodulated and integrated quadratures of the signal.

    """

    input1Bool = Bool(True).tag(pref=True)

    input2Bool = Bool(False).tag(pref=True)

    input1demod= Enum('1','2','3','4','5','6','7','8').tag(pref=True)

    input2demod= Enum('1','2','3','4','5','6','7','8').tag(pref=True)

    AWGcontrol = Bool(False).tag(pref=True)

    pointsnumber = Str('1000').tag(pref=True, feval=VAL_INT)

    average = Bool(True).tag(pref=True)

    powerBool = Bool(False).tag(pref=True)

    Npoints = Str('0').tag(pref=True,feval=VAL_INT)

    database_entries = set_default({'Demod': {}})

    def format_multiple_string(self, string, factor, n):
        s = self.format_and_eval_string(string)
        if isinstance(s, list) or isinstance(s, tuple) or isinstance(s, np.ndarray):
            return [elem*factor for elem in s]
        else:
            return [s*factor]*n

    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super(DemodUHFLITask, self).check(*args,
                                                             **kwargs)

        if not (self.format_and_eval_string(self.pointsnumber) >= 1000):
            test = False
            traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                cleandoc('''At least 1000 points must be recorded. Please make real measurements and not noisy s***.''')

        if not self.input1Bool and not self.input2Bool:
            test = False
            traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                cleandoc('''All measurements are disabled.''')


        return test, traceback

    def perform(self):
        """
        """

        if not self.driver.daq:
            self.start_driver()
        if not self.driver.saveConfig:
            path = self.format_string('{default_path}')
            measId = self.format_string('LabOne_settings_{meas_id}.xml')
            full_path = os.path.join(path,measId)
            self.driver.saveConfiguration(full_path)

        Npoints = self.format_and_eval_string(self.Npoints)
        recordsPerCapture = self.format_and_eval_string(self.pointsnumber)
        demod1=self.format_and_eval_string(self.input1demod)-1
        demod2=self.format_and_eval_string(self.input2demod)-1

        device = self.driver.device

        channel = []

        if self.AWGcontrol:
            self.driver.daq.setInt('/%s/awgs/0/enable' %device, 0)
        self.driver.daq.unsubscribe('/%s/*' % device)
        self.driver.daq.flush()
        if self.input1Bool:
            self.driver.daq.subscribe('/%s/demods/%d/sample' % (device,demod1))
            channel=channel+['1']
        if self.input2Bool:
            self.driver.daq.subscribe('/%s/demods/%d/sample' % (device,demod2))
            channel=channel+['2']

        answerDemod = self.driver.get_demodLI(recordsPerCapture,self.average,Npoints,channel,[demod1,demod2],self.powerBool,self.AWGcontrol)
        self.write_in_database('Demod', answerDemod)

class PulseTransferUHFLITask(InstrumentTask):
    """ Give a pulse sequence to UHFLI's AWG module

    """
    PulseSeqFile = Str('pulseSeqFile.txt').tag(pref=True)
    modified_values = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                    ordered_dict_from_pref))
    mod1 = Bool(False).tag(pref=True)

    mod2 = Bool(False).tag(pref=True)

    autoStart = Bool(True).tag(pref=True)

    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super(PulseTransferUHFLITask, self).check(*args,
                                                             **kwargs)
        file = None
        try:
            file = open(self.PulseSeqFile,"r")
        except FileNotFoundError:
            test = False
            traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                cleandoc('''File not found''')
        if file:
            file.close()

        return test, traceback

    def perform(self):
        """
        """
        if not self.driver.daq:
            self.start_driver()
        if not self.driver.saveConfig:
            path = self.format_string('{default_path}')
            measId = self.format_string('LabOne_settings_{meas_id}.xml')
            full_path = os.path.join(path,measId)
            self.driver.saveConfiguration(full_path)

        device = self.driver.device
        exp_setting = [['/%s/awgs/0/single'               % device, 0]# mode rerun, the awg repeat the sequence
                        ]
        if self.mod1:
            exp_setting.append([ '/%s/awgs/0/outputs/0/mode'  %device, 2])
        else :
            exp_setting.append([ '/%s/awgs/0/outputs/0/mode'  %device, 1])
        if self.mod2:
            exp_setting.append([ '/%s/awgs/0/outputs/1/mode'  %device, 2])
        else :
            exp_setting.append([ '/%s/awgs/0/outputs/1/mode'  %device, 1])

        self.driver.daq.set(exp_setting)
        self.driver.daq.sync()
        if not self.driver.awgModule:
            self.driver.awgModule=self.driver.daq.awgModule()
            self.driver.awgModule.set('awgModule/device', device)
            self.driver.awgModule.execute()

        self.driver.daq.setInt('/%s/awgs/0/enable' %device, 0)
        self.driver.daq.sync()


        file = open(self.PulseSeqFile,"r")
        sequence = file.read()
        file.close()

        awg_program = textwrap.dedent(sequence)

        for l, v in self.modified_values.items():
            v=str(eval_with_units(self,v))
            awg_program = awg_program.replace(l, v)

        self.driver.TransfertSequence(awg_program)

        # Start the AWG in single-shot mode if autoStart is True
        self.driver.daq.setInt('/%s/awgs/0/enable' %device, self.autoStart)
        self.driver.daq.sync()

class SetParametersUHFLITask(InstrumentTask):
    """ Set Lock-in, AWG and Ouput Parameters of UHFLI.

    """
    parameterToSet = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                    ordered_dict_from_pref))


    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super(SetParametersUHFLITask, self).check(*args,
                                                             **kwargs)

#        for p,v in self.parameterToSet.items():
#            if p[:3] == 'Osc':
#                if self.format_and_eval_string(v)<0:
#                    test = False
#                    traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
#                        cleandoc('''Oscillator's frequency musst be positive''')
#            elif p[:3] == 'AWG':
#                if self.format_and_eval_string(v)>1 or self.format_and_eval_string(v)<0:
#                    test = False
#                    traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
#                        cleandoc('''AWG output's amplitude musst be between 0 and 1''')
#            elif p[:3] == 'Use':
#                if self.format_and_eval_string(v)>2**32 or self.format_and_eval_string(v)<0:
#                    test = False
#                    traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
#                        cleandoc('''User Register's value musst be between 0 and 2^32''')

        return test, traceback

    def perform(self):
        """
        """
        if not self.driver.daq:
            self.start_driver()
        if not self.driver.saveConfig:
            path = self.format_string('{default_path}')
            measId = self.format_string('LabOne_settings_{meas_id}.xml')
            full_path = os.path.join(path,measId)
            self.driver.saveConfiguration(full_path)

        device = self.driver.device

        exp_setting=[]

        for p, v in self.parameterToSet.items():
            if v[0][:3] == 'AWG':
                channel =self.format_and_eval_string(v[1])-1
                value = eval_with_units(self,v[-2:])
                exp_setting=exp_setting+ [['/%s/awgs/0/outputs/%d/amplitude' % (device, channel), value]]
            elif v[0][:4] == 'Osci':
                channel =self.format_and_eval_string(v[1])-1
                value = eval_with_units(self,v[-2:])
                exp_setting=exp_setting+ [['/%s/oscs/%d/freq' % (device, channel), value]]
            elif v[0][:3] == 'Pha':
                channel =self.format_and_eval_string(v[1])-1
                value = self.format_and_eval_string(v[-2])
                exp_setting=exp_setting+ [['/%s/demods/%d/phaseshift' % (device, channel), value]]
            elif ' TC ' in v[0]:
                channel = self.format_and_eval_string(v[1])-1
                value = eval_with_units(self,v[-2:])
                exp_setting=exp_setting+ [['/%s/demods/%d/timeconstant' % (device, channel), value]]
            elif ' order' in v[0]:
                channel = self.format_and_eval_string(v[1])-1
                value = self.format_and_eval_string(v[-2])
                exp_setting=exp_setting+ [['/%s/demods/%d/order' % (device, channel), value]]
            elif 'Osc of' in v[0]:
                channel = self.format_and_eval_string(v[1])-1
                value = self.format_and_eval_string(v[-2])-1
                exp_setting=exp_setting+ [['/%s/demods/%d/oscselect' % (device, channel), value]]
            elif 'Output' in v[0]:
                output= self.format_and_eval_string(v[0][6])-1
                channel = self.format_and_eval_string(v[1])-1
                value = eval_with_units(self,v[-2:])
                exp_setting=exp_setting+ [['/%s/sigouts/%d/amplitudes/%d' % (device, output, channel), value]]
            elif 'Trig' in v[0]:
                channel = self.format_and_eval_string(v[1])-1
                if 'High' in v[-2]:
                    value = 0x1000000
                else:
                    value = 0x100000
                value = value*2**(int(v[-2][12])-1)
                exp_setting=exp_setting+ [['/%s/demods/%d/trigger'% (device,channel), value]]
            else :
                channel = self.format_and_eval_string(v[1])-1
                value = eval_with_units(self,v[-2:])
                exp_setting=exp_setting+ [['/%s/awgs/0/userregs/%d' % (device, channel), value]]

        self.driver.daq.set(exp_setting)
        self.driver.daq.sync()

class CloseAWGUHFLITask(InstrumentTask):
    """ Close AWG module of UHFLI.

    """

    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super(CloseAWGUHFLITask, self).check(*args,
                                                             **kwargs)


        return test, traceback

    def perform(self):
        """
        """
        if not self.driver.daq:
            self.start_driver()

        if self.driver.awgModule:
            if self.driver.daq:
                self.driver.daq.setInt('/%s/awgs/0/enable' %self.driver.device, 0)
            self.driver.awgModule.finish()
            self.driver.awgModule.clear()
            self.driver.daq.sync()
            self.driver.awgModule = None

class DAQDemodUHFLITask(InstrumentTask):
    """ Get a trace of the demodulated sigal.
        Can average the trace. Number of channel unlimited.
    """
    delay = Str('0').tag(pref=True)

    numberRow = Str('1').tag(pref=True)

    numberCol = Str('100').tag(pref=True)

    repetition = Str('1').tag(pref=True, feval=VAL_INT)

    numberGrid = Str('1').tag(pref=True)

    average = Bool(True).tag(pref=True)


    AWGControl = Bool(False).tag(pref=True)

    RowRepetition = Bool(False).tag(pref=True)

    trigger = Enum('AWG Trigger 1','AWG Trigger 2','AWG Trigger 3','AWG Trigger 4').tag(pref=True)

    triggerChain = Enum ('Demodulator 1','Demodulator 2','Demodulator 3','Demodulator 4',
                        'Demodulator 5','Demodulator 6','Demodulator 7','Demodulator 8').tag(pref=True)

    signalDict = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                    ordered_dict_from_pref))


    database_entries = set_default({'Grid': {}})


    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super(DAQDemodUHFLITask, self).check(*args,
                                                             **kwargs)

        numberRow=self.format_and_eval_string(self.numberRow);
        numberCol=self.format_and_eval_string(self.numberCol);
        numberGrid=self.format_and_eval_string(self.numberGrid);


        if 0 == numberRow or 0 == numberCol or 0 == numberGrid:
            test = False
            traceback[self.task_path + '/' + self.task_name + '-get_grid'] = \
                           cleandoc('''Acquisition musst at least contains 1 row, 1 col and 1 grid''')


        if len(self.signalDict.keys()) == 0 :
            test = False
            traceback[self.task_path + '/' + self.task_name + '-get_grid'] = \
                           cleandoc('''At least select one path for a signal to acquire''')

        return test, traceback

    def perform(self):
        """
        """
        if not self.driver:
            self.start_driver()
        if not self.driver.saveConfig:
            path = self.format_string('{default_path}')
            measId = self.format_string('LabOne_settings_{meas_id}.xml')
            full_path = os.path.join(path,measId)
            self.driver.saveConfiguration(full_path)

        self.driver.set_general_setting();

        delay = self.format_and_eval_string(self.delay)*10**-9
        numberGrid = self.format_and_eval_string(self.numberGrid)
        numberRow = self.format_and_eval_string(self.numberRow)
        numberCol = self.format_and_eval_string(self.numberCol)
        repetition = self.format_and_eval_string(self.repetition)

        device = self.driver.device

        signal_paths=[]
        signalID=[]
        if self.average:
            avg='.avg'
        else:
            avg=''
        for d, s in self.signalDict.items():
            signal_paths.append('/%s/demods/%d/sample.%s'%(device,int(s[0][-1])-1,s[1])+avg)
            self.driver.daq.setInt('/%s/demods/%d/enable' % (device, int(s[0][-1])-1), 1)
            signalID.append(s[0][-1]+s[1])
        self.driver.daq.sync()
        #give experimental setting

        exp_setting = [['dataAcquisitionModule/device' , device],
                        ['dataAcquisitionModule/grid/mode', 4],# grid mode = exact
                        ['dataAcquisitionModule/grid/cols', numberCol],
                        ['dataAcquisitionModule/grid/rows', numberRow],
                        ['dataAcquisitionModule/count',numberGrid],
                        ['dataAcquisitionModule/triggernode','/%s/demods/%d/sample.TrigAWGTrig%s' %(device,int(self.triggerChain[-1])-1,self.trigger[-1])],
                        ['dataAcquisitionModule/clearhistory', 1],
                        ['dataAcquisitionModule/type', 6],
                        ['dataAcquisitionModule/delay', delay],
                        ['dataAcquisitionModule/enable',1],
                        ['dataAcquisitionModule/holdoff/time', 0],
                        ['dataAcquisitionModule/holdoff/count', 0],
                        ['dataAcquisitionModule/grid/rowrepetition', int(self.RowRepetition)],
                        ['dataAcquisitionModule/grid/repetitions', repetition],
                        ['dataAcquisitionModule/awgcontrol', int(self.AWGControl)]]

        DAM = self.driver.daq.dataAcquisitionModule();

        self.driver.daq.unsubscribe('/dev2375/*')
        self.driver.daq.flush()
        DAM.unsubscribe('/%s/*' %device)
        for signal_path in signal_paths:
            DAM.subscribe(signal_path)

        DAM.set(exp_setting)
        self.driver.daq.sync()

        answerDAM = self.driver.get_DAQmodule(DAM,[numberGrid,numberRow,numberCol,repetition],
                                               signalID,signal_paths)

        self.write_in_database('Grid', answerDAM)
