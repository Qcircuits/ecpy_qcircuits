# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by exopyHqcLegacy Authors, see AUTHORS for more details.
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

from exopy.tasks.api import InstrumentTask, validators
from exopy.utils.atom_util import ordered_dict_from_pref, ordered_dict_to_pref

VAL_REAL = validators.Feval(types=numbers.Real)

VAL_INT = validators.Feval(types=numbers.Integral)# -*- coding: utf-8 -*-
        
class PulseTransferHDAWGTask(InstrumentTask):
    """ Give a pulse sequence to HDAWG
        
    """
    PulseSeqFile = Unicode('pulseSeqFile.txt').tag(pref=True)
    
    modified_values = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                    ordered_dict_from_pref))
# =============================================================================
#     mod1 = Bool(False).tag(pref=True)
# 
#     mod2 = Bool(False).tag(pref=True)
# =============================================================================

    
# =============================================================================
#     def check(self, *args, **kwargs):
#         """
#         """
#         test, traceback = super(PulseTransferUHFLITask, self).check(*args,
#                                                              **kwargs)
#         file = None      
#         try:
#             file = open(self.PulseSeqFile,"r")
#         except FileNotFoundError as er:
#             test = False
#             traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
#                 cleandoc('''File not found''')
#         if file:
#             file.close()
#        
#         return test, traceback
# =============================================================================
    
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
        exp_setting = [['/%s/awgs/0/single' % device, 0]]# mode rerun, the awg repeat the sequence]


        self.driver.daq.set(exp_setting)
        self.driver.daq.sync()
        if not self.driver.awgModule:
            self.driver.awgModule = self.driver.daq.awgModule()
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
        self.driver.daq.setInt('/%s/awgs/0/enable' %device, 1) # start/stop equiv
        for channel in self.driver.channels:    
            self.driver.daq.setInt('/{}/sigouts/{}/on'.format(device,channel), 1) # channel on
        self.driver.daq.sync()
        
class SetParametersHDAWGTask(InstrumentTask):
    """ Set Lock-in, AWG and Ouput Parameters of UHFLI.
        
    """    
    parameterToSet = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                    ordered_dict_from_pref))
    
    
#    def check(self, *args, **kwargs):
#        """
#        """
#        test, traceback = super(SetParametersUHFLITask, self).check(*args,
#                                                             **kwargs)
#        
#        for p,v in self.parameterToSet.items():
#            if p[:3] == 'Osc':
#                if self.format_and_eval_string(v)>600e6:
#                    test = False
#                    traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
#                        cleandoc('''Oscillator's frequency must be lower than 600MHz''')
#                if self.format_and_eval_string(v)<0:
#                    test = False
#                    traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
#                        cleandoc('''Oscillator's frequency must be positive''')
#            elif p[:3] == 'AWG':
#                if self.format_and_eval_string(v)>1 or self.format_and_eval_string(v)<0:
#                    test = False
#                    traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
#                        cleandoc('''AWG output's amplitude must be between 0 and 1''')
#            elif p[:3] == 'Use':
#                if self.format_and_eval_string(v)>2**32 or self.format_and_eval_string(v)<0:
#                    test = False
#                    traceback[self.task_path + '/' + self.task_name + '-get_demod'] = \
#                        cleandoc('''User Register's value must be between 0 and 2^32''')
#                        
#        return test, traceback
    
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
        #self.driver.daq.getInt('/%s/system/awg/channelgrouping' %device) 0 = 4x2; 1 = 2x4; 2 = 1x8
        ch_group = 2**(self.driver.daq.getInt('/%s/system/awg/channelgrouping' %device)+1) #number of channels per group
        
        exp_setting=[]
#        ['User Register', 'Oscillator', 'Waveform Amplitude','Phase shift',
#                   'LowPass order','LowPass TC','Osc of Demod ','Trig of Demod',
#                   'Output1 Demod','Output2 Demod']
        for p, v in self.parameterToSet.items():
            if p[:-1].rstrip() == 'Waveform Amplitude':
                channel = self.format_and_eval_string(p[-1])-1
                value = self.format_and_eval_string(v)
                awg = channel//ch_group
                channel = channel%ch_group
                exp_setting = exp_setting + [['/%s/awgs/%d/outputs/%d/amplitude' % (device,awg,channel), value]]
            elif p[:-1].rstrip() == 'Oscillator':
                channel =self.format_and_eval_string(p[-1])-1
                value = self.format_and_eval_string(v)
                exp_setting = exp_setting + [['/%s/oscs/%d/freq' % (device, channel), value]]
            elif p[:-1].rstrip() == 'Phase shift':
                channel =self.format_and_eval_string(p[-1])-1
                value = self.format_and_eval_string(v)
                exp_setting=exp_setting + [['/%s/sines/%d/phaseshift' % (device, channel), value]]
            elif p[:-1].rstrip() == 'User Register':
                channel =self.format_and_eval_string(p[-1])-1
                value = self.format_and_eval_string(v)
                exp_setting=exp_setting + [['/%s/awgs/0/userregs/%d' % (device, channel), value]]
            elif p[:-1].rstrip() == 'Amplitude Range':
                channel = self.format_and_eval_string(p[-1])-1
                value = self.format_and_eval_string(v)
                exp_setting=exp_setting + [['/%s/sigouts/%d/range' % (device, channel), value]]
            else:
                print('not an interfaced variable')
                
#            elif ' TC ' in p:
#                channel = self.format_and_eval_string(p[-1])-1
#                value = self.format_and_eval_string(v)
#                exp_setting=exp_setting + [['/%s/demods/%d/timeconstant' % (device, channel), value]]
#            elif ' order' in p:
#                channel = self.format_and_eval_string(p[-1])-1
#                value = self.format_and_eval_string(v)
#                exp_setting=exp_setting + [['/%s/demods/%d/order' % (device, channel), value]]
#            elif 'Osc of' in p:
#                channel = self.format_and_eval_string(p[-1])-1
#                value = self.format_and_eval_string(v)-1
#                exp_setting=exp_setting + [['/%s/demods/%d/oscselect' % (device, channel), value]]
#            elif 'Output' in p:
#                output= self.format_and_eval_string(p[6])-1
#                channel = self.format_and_eval_string(p[-1])-1
#                value = self.format_and_eval_string(v)
#                exp_setting=exp_setting + [['/%s/sigouts/%d/amplitudes/%d' % (device, output, channel), value]]
#            elif 'Trig' in p:
#                channel = self.format_and_eval_string(p[-1])-1
#                if 'High' in v:
#                    value = 0x1000000
#                else:
#                    value = 0x100000
#                value = value*2**(int(v[12])-1)
#                exp_setting=exp_setting + [['/%s/demods/%d/trigger'% (device,channel), value]]
#            else :
#                if p[-2]=='1':
#                    channel = self.format_and_eval_string(p[-2:])-1
#                else:
#                    channel = self.format_and_eval_string(p[-1])-1
#                value = self.format_and_eval_string(v)
#                exp_setting=exp_setting + [['/%s/awgs/0/userregs/%d' % (device, channel), value]]
                
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
        