# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by exopyHqcLegacy Authors, see AUTHORS for more details.
#
# DiUnicodeibuted under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Task perform measurements for the UFHLI zurich instruments.

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

def eval_with_units(task,evaluee):
    value = task.format_and_eval_string(evaluee[0])
    unit = str(evaluee[1])
    unitlist = ['','none', 'ns', 'GHz' ,'clock_samples', 's', 'µs','ns_to_clck'] #from views
    multlist  = [1,1,1e-9,1e9,1,1,1e-6,1/3.33] #corresponding to above list
    unitdict = dict(zip(unitlist,multlist))

    clckrate_command = '/%s/system/clocks/sampleclock/freq' %task.driver.device
    unitdict['ns_to_clck'] = float(task.driver.daq.get(clckrate_command, True, 0)[clckrate_command]['value'])*1e-9/8 # the HDAWG clockrate is 8 times slower than the sample rate
    
    multiplier = unitdict[unit]
    value = value * multiplier
    return value
        
class PulseTransferHDAWGTask(InstrumentTask):
    """ Give a pulse sequence to HDAWG
        
    """
    PulseSeqFile = Unicode('pulseSeqFile.txt').tag(pref=True)
    
    modified_values = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                    ordered_dict_from_pref))
    reference_values = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                    ordered_dict_from_pref))
    
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
            value = str(eval_with_units(self,v))
            awg_program = awg_program.replace(l, value)        
        
        self.driver.TransferSequence(awg_program)

        # Start the AWG in single-shot mode. CAREFUL this does not activate the outputs
        self.driver.daq.setInt('/%s/awgs/0/enable' %device, 1) # start/stop equiv
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
            if v[0] == 'Waveform Amplitude':
                channel = self.format_and_eval_string(v[1])-1
                value = eval_with_units(self,v[-2:])
                awg = channel//2 #2 channels per AWG
                channel = channel%2
                exp_setting = exp_setting + [['/%s/awgs/%d/outputs/%d/amplitude' % (device,awg,channel), value]]
            elif v[0] == 'Oscillator':
                channel = self.format_and_eval_string(v[1])-1
                value = eval_with_units(self,v[-2:])
                exp_setting = exp_setting + [['/%s/oscs/%d/freq' % (device, channel), value]]
            elif v[0] == 'Phase shift':
                channel =self.format_and_eval_string(v[1])-1
                value = eval_with_units(self,v[-2:])
                exp_setting=exp_setting + [['/%s/sines/%d/phaseshift' % (device, channel), value]]
            elif v[0] == 'User Register':
                channel =self.format_and_eval_string(v[1])-1
                value = eval_with_units(self,v[-2:])
                exp_setting=exp_setting + [['/%s/awgs/0/userregs/%d' % (device, channel), np.floor(value)]]
            elif v[0] == 'Amplitude Range':
                channel = self.format_and_eval_string(v[1])-1
                value = eval_with_units(self,v[-2:])
                exp_setting=exp_setting + [['/%s/sigouts/%d/range' % (device, channel), value]]
            elif v[0] == 'Hold':
                channel = self.format_and_eval_string(v[1])-1
                value = int(bool(v[2]=='On'))
                awg = channel//2 #2 channels per AWG
                channel = channel%2
                exp_setting=exp_setting + [['/%s/awgs/%d/outputs/%d/hold' % (device,awg,channel), value]]
            else:
                print('not an interfaced variable')
                                
        self.driver.daq.set(exp_setting)
        self.driver.daq.sync()

class OutputOnOffHDAWGTask(InstrumentTask):
    
    channellist = Unicode().tag(pref=True)
    onoff = Unicode().tag(pref=True)
    
    def perform(self):
        if not self.driver.daq:
            self.start_driver()
        
        if not self.driver.saveConfig:
            path = self.format_string('{default_path}')
            measId = self.format_string('LabOne_settings_{meas_id}.xml')
            full_path = os.path.join(path,measId)
            self.driver.saveConfiguration(full_path)
        
        device = self.driver.device        
        channels = list(map(lambda x: int(x)-1,self.channellist.split(',')))
        setOn = 0
        if self.onoff in ['On','on','1']:
            setOn = 1
        elif self.onoff in ['Off','off','0']:
            setOn = 0
        else:
            print('Invalid on/off string')
            raise ValueError
    
        for ch in channels: 
            self.driver.daq.setInt('/{}/sigouts/{}/on'.format(device,ch), setOn)