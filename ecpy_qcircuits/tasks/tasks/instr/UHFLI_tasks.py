# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by EcpyHqcLegacy Authors, see AUTHORS for more details.
#
# DiUnicodeibuted under the terms of the BSD license.
#
# The full license is in the file LICENCE, diUnicodeibuted with this software.
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

from atom.api import (Bool, Unicode, Enum, set_default,Typed)

from ecpy.tasks.api import InstrumentTask, validators
from ecpy.utils.atom_util import ordered_dict_from_pref, ordered_dict_to_pref


VAL_REAL = validators.Feval(types=numbers.Real)

VAL_INT = validators.Feval(types=numbers.Integral)# -*- coding: utf-8 -*-

class ScopeDemodUHFLITask(InstrumentTask):
    """ Get the raw or averaged quadratures of the signal.
        Can also get raw or averaged traces of the signal.
    """
    freq = Unicode('25').tag(pref=True)
    
    freq2 = Unicode('25').tag(pref=True)

    delay = Unicode('0').tag(pref=True)
    
    delay2 = Unicode('0').tag(pref=True)

    duration = Unicode('1000').tag(pref=True)
    
    duration2 = Unicode('0').tag(pref=True)

    tracesnumber = Unicode('1000').tag(pref=True, feval=VAL_INT)

    average = Bool(True).tag(pref=True)
    
    demod = Bool(True).tag(pref=True)

    trace = Bool(False).tag(pref=True)    
    
    demod2 = Bool(True).tag(pref=True)

    trace2 = Bool(False).tag(pref=True)
    
    Npoints = Unicode('1').tag(pref=True,feval=VAL_INT)
    
    customDemod = Unicode('[]').tag(pref=True)
    
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
                        ['/%s/scopes/0/trigreference'   % device, 0]]

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

    pulse_duration1 = Unicode('1000').tag(pref=True)

    pulse_duration2 = Unicode('0').tag(pref=True)

    pointsnumber = Unicode('1000').tag(pref=True, feval=VAL_INT)

    average = Bool(True).tag(pref=True)
            
    sequence_duration = Unicode('30000').tag(pref=True)
    
    Npoints = Unicode('0').tag(pref=True,feval=VAL_INT)
        
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

        sequence_duration = self.format_multiple_string(self.sequence_duration, 10**-9, 1)

        if 0 in sequence_duration:
            test = False
            traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                cleandoc('''All measurements are disabled.''')
        
        pulse_duration1 = self.format_multiple_string(self.pulse_duration1, 10**-9, 1)
        if pulse_duration1 > sequence_duration:
            test = False
            traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                cleandoc('''The pulse duration of the channel 1 should be smaller than the sequence duration''')
                
        pulse_duration2 = self.format_multiple_string(self.pulse_duration2, 10**-9, 1)
        if pulse_duration2 > sequence_duration:
            test = False
            traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                cleandoc('''The pulse duration of the channel 2 should be smaller than the sequence duration''')
               
        
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
        pulse_duration1 = self.format_and_eval_string(self.pulse_duration1)*10**-9
        pulse_duration2 = self.format_and_eval_string(self.pulse_duration2)*10**-9
        sequence_duration = self.format_and_eval_string(self.sequence_duration)*10**-9
        #give experimental setting
        # the  program is made to work with the demodulator 4 and the oscillators 1 and 7
        self.driver.set_general_setting();

        device = self.driver.device
        # le demodulateur 4 est indexé 3
        exp_setting = [['/%s/demods/3/timeconstant'    % device, pulse_duration1/3], # time constant of the filter
                       ['/%s/demods/2/timeconstant'    % device, pulse_duration2/3], 
                       ['/%s/oscs/6/freq'              % device, 1/sequence_duration],
                       ['/%s/demods/6/oscselect'       % device, 6]#demodulator 7 use oscillator 7
                      ]
        channel = []
        self.driver.daq.set(exp_setting)      
        self.driver.daq.unsubscribe('/%s/*' % device)
        self.driver.daq.flush()
        if not pulse_duration1 ==0:
            self.driver.daq.subscribe('/%s/demods/3/sample' % device)
            channel=channel+['1']
        if not pulse_duration2 ==0:
            self.driver.daq.subscribe('/%s/demods/2/sample' % device)
            channel=channel+['2']
        self.driver.daq.sync()
        # Perform a global synchronisation between the device and the data server:
        # Ensure that the settings have taken effect on the device before acquiring
        # data.    
        answerDemod = self.driver.get_demodLI(recordsPerCapture,self.average,Npoints,channel)
        self.write_in_database('Demod', answerDemod)
        
class PulseTransferUHFLITask(InstrumentTask):
    """ Give a pulse sequence to UHFLI's AWG module
        
    """
    PulseSeqFile = Unicode('pulseSeqFile.txt').tag(pref=True)
    
    modified_values = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                    ordered_dict_from_pref))
    mod1 = Bool(False).tag(pref=True)

    mod2 = Bool(False).tag(pref=True)

    
    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super(PulseTransferUHFLITask, self).check(*args,
                                                             **kwargs)
        file = None      
        try:
            file = open(self.PulseSeqFile,"r")
        except FileNotFoundError as er:
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
            v=str(self.format_and_eval_string(v))
            awg_program = awg_program.replace(l, v)
        
        self.driver.TransfertSequence(awg_program)

        # Start the AWG in single-shot mode.
        self.driver.daq.setInt('/%s/awgs/0/enable' %device, 1)
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
        
        for p,v in self.parameterToSet.items():
            if p[:3] == 'Osc':
                if self.format_and_eval_string(v)>600e6:
                    test = False
                    traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                        cleandoc('''Oscillator's frequency musst be lower than 600MHz''')
                if self.format_and_eval_string(v)<0:
                    test = False
                    traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                        cleandoc('''Oscillator's frequency musst be positive''')
            elif p[:3] == 'AWG':
                if self.format_and_eval_string(v)>1 or self.format_and_eval_string(v)<0:
                    test = False
                    traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                        cleandoc('''AWG output's amplitude musst be between 0 and 1''')
            elif p[:3] == 'Use':
                if self.format_and_eval_string(v)>2**32 or self.format_and_eval_string(v)<0:
                    test = False
                    traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
                        cleandoc('''User Register's value musst be between 0 and 2^32''')
                        
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
            if p[:3] == 'AWG':
                channel =self.format_and_eval_string(p[-1])-1
                value = self.format_and_eval_string(v)
                exp_setting=exp_setting+ [['/%s/awgs/0/outputs/%d/amplitude' % (device, channel), value]]
            elif p[:4] == 'Osci':
                channel =self.format_and_eval_string(p[-1])-1
                value = self.format_and_eval_string(v)
                exp_setting=exp_setting+ [['/%s/oscs/%d/freq' % (device, channel), value]]
            elif p[:3] == 'Pha':
                channel =self.format_and_eval_string(p[-1])-1
                value = self.format_and_eval_string(v)
                exp_setting=exp_setting+ [['/%s/demods/%d/phaseshift' % (device, channel), value]]
            elif ' TC ' in p:
                channel = self.format_and_eval_string(p[-1])-1
                value = self.format_and_eval_string(v)
                exp_setting=exp_setting+ [['/%s/demods/%d/timeconstant' % (device, channel), value]]
            elif ' order' in p:
                channel = self.format_and_eval_string(p[-1])-1
                value = self.format_and_eval_string(v)
                exp_setting=exp_setting+ [['/%s/demods/%d/order' % (device, channel), value]]
            elif 'Osc of' in p:
                channel = self.format_and_eval_string(p[-1])-1
                value = self.format_and_eval_string(v)-1
                exp_setting=exp_setting+ [['/%s/demods/%d/oscselect' % (device, channel), value]]
            elif 'Output' in p:
                output= self.format_and_eval_string(p[6])-1
                channel = self.format_and_eval_string(p[-1])-1
                value = self.format_and_eval_string(v)
                exp_setting=exp_setting+ [['/%s/sigouts/%d/amplitudes/%d' % (device, output, channel), value]]
            elif 'Trig' in p:
                channel = self.format_and_eval_string(p[-1])-1
                if 'High' in v:
                    value = 0x1000000
                else:
                    value = 0x100000
                value = value*2**(int(v[12])-1)
                exp_setting=exp_setting+ [['/%s/demods/%d/trigger'% (device,channel), value]]
            else :
                if p[-2]=='1':
                    channel = self.format_and_eval_string(p[-2:])-1
                else:
                    channel = self.format_and_eval_string(p[-1])-1
                value = self.format_and_eval_string(v)
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
    delay = Unicode('0').tag(pref=True)
    
    numberRow = Unicode('1').tag(pref=True)
    
    numberCol = Unicode('100').tag(pref=True)

    repetition = Unicode('1').tag(pref=True, feval=VAL_INT)

    numberGrid = Unicode('1').tag(pref=True)

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
        numberGrid=self.format_and_eva_stringl(self.numberGrid);


        if 0 == numberRow or 0 == numberCol or 0 == numberGrid:
            test = False
            traceback[self.task_path + '/' + self.task_name + '-get_grid'] = \
                           cleandoc('''Acquisition musst at least contains 1 row, 1 col and 1 grid''')


        if len(self.signalDict.keys()) == 0 :
            test = False
            traceback[self.task_path + '/' + self.task_name + '-get_grid'] = \
                           cleandoc('''At least select one pass for a signal to acquire''')
        
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
            signal_paths.append('/%s/demods/%d/sample.%s'+avg % (device,int(d[-1])-1,s))
            self.driver.daq.setInt('/%S/demods/%d/enable' % (device, int(d[-1])-1), 1)
            signalID.append([d[-1]+s+avg])
        self.driver.daq.sync()
        #give experimental setting

        exp_setting = [['dataAcquisitionModule/device' , device],
                        ['dataAcquisitionModule/grid/mode', 4],# grid mode = exact
                        ['dataAcquisitionModule/grid/cols', numberCol],
                        ['dataAcquisitionModule/grid/rows', numberRow],
                        ['dataAcquisitionModule/count',numberGrid]
                        ['dataAcquisitionModule/triggernode', 
                         '/%s/demods/%s/sample.TrigAWGTrig%s' %(device,self.triggerChain[-1],self.trigger[-1])],
                        ['dataAcquisitionModule/clearhistory', 1],
                        ['dataAcquisitionModule/type', 6],
                        ['dataAcquisitionModule/delay', delay],
                        ['dataAcquisitionModule/enable',1],
                        ['dataAcquisitionModule/holdoff/time', 0],
                        ['dataAcquisitionModule/holdoff/count', 0],
                        ['dataAcquisitionModule/grid/rowrepetition', int(self.RowRepetition)],
                        ['dataAcquisitionModule/grid/repetitions', repetition]]
       
        DAM = self.driver.daq.dataAcquisitionModule();
        DAM.set(exp_setting)
        self.driver.daq.unsubscribe('/dev2375/*')
        self.driver.daq.flush()
        DAM.unsubscribe('/dev2375/*')
        for signal_path in signal_paths:
            DAM.subscribe(signal_path)
        self.driver.daq.sync()
        
        answerDAM = self.driver.get_DAQmodule(DAM,[numberCol,numberRow,numberCol],
                                               signalID,signal_paths)

        self.write_in_database('Grid', answerDAM)
        
        
        