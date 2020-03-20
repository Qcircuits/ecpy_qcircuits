# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by exopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Task perform measurements for the HDAWG zurich instruments.

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
from operator import mul
from functools import reduce

from atom.api import (Bool, Str, Enum, set_default,Typed)

from exopy.tasks.api import InstrumentTask, validators
from exopy.utils.atom_util import ordered_dict_from_pref, ordered_dict_to_pref

VAL_REAL = validators.Feval(types=numbers.Real)

VAL_INT = validators.Feval(types=numbers.Integral)# -*- coding: utf-8 -*-

def eval_with_units(task,evaluee):
    value = task.format_and_eval_string(evaluee[0])
    unit = str(evaluee[1])
    unitlist = ['','none', 'ns', 'GHz', 'MHz' ,'clock_samples', 's', 'µs','ns_to_clck'] #from views
    multlist  = [1,1,1e-9,1e9,1e6,1,1,1e-6,1/3.33] #corresponding to above list
    unitdict = dict(zip(unitlist,multlist))

    clckrate_command = '/%s/system/clocks/sampleclock/freq' %task.driver.device
    unitdict['ns_to_clck'] = float(task.driver.daq.get(clckrate_command, True, 0)[clckrate_command]['value'])*1e-9/8 # the HDAWG clockrate is 8 times slower than the sample rate

    multiplier = unitdict[unit]
    value = value * multiplier
    return value

class PulseTransferHDAWGTask(InstrumentTask):
    """ Give a pulse sequence to HDAWG

    """
    PulseSeqFile = Str('pulseSeqFile.txt').tag(pref=True)

    modified_values = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                    ordered_dict_from_pref))
    reference_values = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                    ordered_dict_from_pref))
           
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
        exp_setting = [['/%s/awgs/0/single' % device, 1]]# mode rerun, the awg repeat the sequence]


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
        
        # replace text keys with values from exopy
        for l, v in self.modified_values.items():
            value = str(eval_with_units(self,v))
            awg_program = awg_program.replace(l, value)
           
        # transfer the sequence to the AWG
        self.driver.TransferSequence(awg_program)
        
        # execute commented commands in sequence from exopy if not already implemented
        #replace_vals = {}
        for l, v in self.modified_values.items():
            value = str(eval_with_units(self,v))
            #replace_vals[l[1:-1]] = value
            awg_program = awg_program.replace(l[1:-1], value)
            
        lines = awg_program.split('\n')
        commands = []
        for line in lines:
            if line[:7] == '//CMND ':
                commands.append(line[7:-1])
        
        command_strings = []
        command_prefixes = []
        command_values = []
        for i,cmnd in enumerate(commands):
            command_parts = cmnd.split('"')
            cmnd_str = '/%s/'%device + command_parts[1] + ''
            command_strings.append(cmnd_str)
            command_prefixes.append(command_parts[0][:-1])
            command_values.append(command_parts[2].replace(" ", "")[1:-1])
            command_parts[1] = cmnd_str
        
        for i,cmnd in enumerate(command_strings):
            if command_prefixes[i] == 'setInt':
                val = self.driver.daq.getInt(cmnd)
                if abs(val-self.format_and_eval_string(command_values[i])) > 0.1:
                    self.driver.daq.setInt(cmnd,self.format_and_eval_string(command_values[i]))
            elif command_prefixes[i] == 'setDouble':
                val = self.driver.daq.getDouble(cmnd)
                if abs(val-self.format_and_eval_string(command_values[i])) > 0.001:
                    self.driver.daq.setDouble(cmnd,self.format_and_eval_string(command_values[i]))
            else:
                raise ValueError('The sequence command was not recognised...')
        self.driver.daq.sync()
        
        

        # Start the AWG in single-shot mode. CAREFUL this does not activate the outputs
        self.driver.daq.setInt('/%s/awgs/0/enable' %device, 1) # start/stop equiv
        self.driver.daq.sync()
        
        time.sleep(0.5) #gives HDAWG time to send commands to various output settings (oscillators etc etc)
        
        #now wait for the signal outputs to set the correct delays
        while not reduce(mul, [1-self.driver.daq.getInt('/{}/sigouts/{}/busy'.format(device,i)) for i in self.driver.channels], 1):
            time.sleep(0.5)
            
        self.driver.daq.sync()
        
class SetParametersHDAWGTask(InstrumentTask):
    """ Set Lock-in, AWG and Ouput Parameters of UHFLI.
        
    """    
    parameterToSet = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                    ordered_dict_from_pref))
   
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
        #ch_group = 2**(self.driver.daq.getInt('/%s/system/awg/channelgrouping' %device)+1) #number of channels per group
        
        exp_setting = []

        wait_after_sync = False
        for p, v in self.parameterToSet.items():
            
            if v[0] == 'Waveform Amplitude':
                channel = self.format_and_eval_string(v[1])-1
                wave = self.format_and_eval_string(v[2])-1
                value = self.format_and_eval_string(v[3])
                awg = channel//2 #2 channels per AWG
                channel = channel%2
                wave = wave%2
                command = '/%s/awgs/%d/outputs/%d/gains/%d'
                current_val = self.driver.daq.getDouble(command % (device,awg,channel,wave))
                if abs(current_val - value) > 0.001:
                    exp_setting = exp_setting + [[command % (device,awg,channel,wave), value]]
                    
            elif v[0] == 'Oscillator':
                channel = self.format_and_eval_string(v[1])-1
                value = eval_with_units(self,v[-2:])
                command = '/%s/oscs/%d/freq'
                current_val = self.driver.daq.getDouble(command % (device,channel))
                if abs(current_val - value) > 1: #1 Hz precision
                    exp_setting = exp_setting + [[command % (device,channel), value]]
                    
            elif v[0] == 'Phase shift':
                channel =self.format_and_eval_string(v[1])-1
                value = self.format_and_eval_string(v[2])
                command = '/%s/sines/%d/phaseshift'
                current_val = self.driver.daq.getDouble(command % (device,channel))
                if abs(current_val - value) > 0.01:
                    exp_setting = exp_setting + [[command % (device,channel), value]]
                    
            elif v[0] == 'User Register':
                channel = self.format_and_eval_string(v[1])-1
                value = eval_with_units(self,v[-2:])
                command = '/%s/awgs/0/userregs/%d'
                current_val = self.driver.daq.getInt(command % (device,channel))
                if current_val != value:
                    exp_setting = exp_setting + [[command % (device,channel), value]]
                    
            elif v[0] == 'Amplitude Range':
                channel = self.format_and_eval_string(v[1])-1
                value = eval_with_units(self,v[-2:])
                command = '/%s/sigouts/%d/range'
                current_val = self.driver.daq.getDouble(command % (device,channel))
                if abs(current_val - value) > 0.01:
                    exp_setting = exp_setting + [[command % (device,channel), value]]
                    
            elif v[0] == 'Hold':
                channel = self.format_and_eval_string(v[1])-1
                value = int(bool(v[2]=='On'))
                awg = channel//2 #2 channels per AWG
                channel = channel%2
                command = '/%s/awgs/%d/outputs/%d/hold'
                current_val = self.driver.daq.getInt(command % (device,awg,channel))
                if current_val != value:
                    exp_setting = exp_setting + [[command % (device,awg,channel), value]]
                    
            elif v[0] == 'Modulation Mode':
                channel = self.format_and_eval_string(v[1])-1
                if v[2] == 'Off':
                    value = 0
                elif v[2] in ['11','33','55','77']:
                    value = 1
                elif v[2] in ['22','44','66','88']:
                    value = 2
                elif v[2] in ['12','34','56','78']:
                    value = 3
                elif v[2] in ['21','43','65','87']:
                    value = 4
                elif v[2] == 'Advanced':
                    value = 5
                else:
                    print('Modulation mode not supported')
                    raise ValueError
                
                command = '/{}/awgs/{}/outputs/{}/modulation/mode'.format(device, (int(v[1])-1)//2, (int(v[1])-1)%2)
                current_val = self.driver.daq.getInt(command)
                if current_val != value:
                    exp_setting = exp_setting + [[command, value]]
                
                wait_after_sync = True

            else:
                print('not an interfaced variable')

        self.driver.daq.set(exp_setting)
        self.driver.daq.sync()

        if wait_after_sync: #sometimes we need to wait a bit for the AWG cores to synchronise
            time.sleep(0.5)
            while not reduce(mul, [1-self.driver.daq.getInt('/{}/sigouts/{}/busy'.format(device,i)) for i in self.driver.channels], 1):
                time.sleep(0.5)

        self.driver.daq.sync()
        
class OutputOnOffHDAWGTask(InstrumentTask):

    channellist = Str().tag(pref=True)
    onoff = Str().tag(pref=True)

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
