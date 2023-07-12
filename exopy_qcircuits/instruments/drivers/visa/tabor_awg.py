# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2016 by ExopyHqcLegacy and ExopyQcircuits Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
""" This module defines drivers for Tabor AWG using VISA library.

:Contains:
    TaborAWGChannel
    TaborAWG


"""
from threading import Lock
from contextlib import contextmanager
from ..driver_tools import (BaseInstrument, InstrIOError, secure_communication,
                            instrument_property)
from ..visa_tools import VisaInstrument
from pyvisa import VisaTypeError
from textwrap import fill
from inspect import cleandoc
import re
import time
import numpy as np


class TaborAWGChannel(BaseInstrument):

    def __init__(self, AWG, channel_num, caching_allowed=True,
                 caching_permissions={}):
        super(TaborAWGChannel, self).__init__(None, caching_allowed,
                                         caching_permissions)
        self._AWG = AWG
        self._channel = channel_num
        AWG.write_termination = '\n'
        AWG.read_termination = '\n'

    def reopen_connection(self):
        self._AWG.reopen_connection()


    @contextmanager
    def secure(self):
        """ Lock acquire and release method

        """
        i = 0
        while not self._AWG.lock.acquire():
            time.sleep(0.1)
            i += 1
            if i > 50:
                raise InstrIOError
        try:
            yield
        finally:
            self._AWG.lock.release()

    @instrument_property
    @secure_communication()
    def output_state(self):
        """ Output getter method

        """
        with self.secure():
            self._AWG.write('INST {}'.format(self._channel))
            output = self._AWG.ask('OUTP?')
            if output == 'ON':
                return 'ON'
            elif output == 'OFF':
                return 'OFF'
            else:
                mes = cleandoc('AWG channel {} did not return its output'
                               .format(self._channel))
                raise InstrIOError(mes)

    @output_state.setter
    @secure_communication()
    def output_state(self, value):
        """ Output setter method. 'ON', 'OFF'

        """
        with self.secure():
            on = re.compile('on', re.IGNORECASE)
            off = re.compile('off', re.IGNORECASE)
            if on.match(value) or value == 1:

                self._AWG.write('INST {}'.format(self._channel))
                self._AWG.write('SOUR:FUNC:MODE USER')
                self._AWG.write('OUTP ON')
                if self._AWG.ask('OUTP?'
                                            ) != 'ON':
                    raise InstrIOError(cleandoc('''Instrument did not set
                                                correctly the output'''))
            elif off.match(value) or value == 0:
                self._AWG.write('INST {}'.format(self._channel))
                self._AWG.write('SOUR:FUNC:MODE USER')
                self._AWG.write('OUTP OFF')
                if self._AWG.ask('OUTP?'
                                            ) != 'OFF':
                    raise InstrIOError(cleandoc('''Instrument did not set
                                                correctly the output'''))
            else:
                mess = fill(cleandoc('''The invalid value {} was sent to
                            switch_on_off method''').format(value), 80)
                raise VisaTypeError(mess)
    
    @secure_communication()        
    def output_mode(self):
        """ Output mode getter method

        """
        with self.secure():
            self._AWG.write('INST {}'.format(self._channel))
            if self.output_state() == 'OFF':
                mes = cleandoc('AWG channel {} is not currently outputting'
                               .format(self._channel))
                raise InstrIOError(mes)
            else:
                mode = self._AWG.ask('SOUR:FUNC:MODE?')
                return mode
            
    @secure_communication()   
    def set_output_mode(self,value):
        """ Output mode setter method

        """
        with self.secure():
            self._AWG.write('INST {}'.format(self._channel))
            if self.output_state() == 'OFF':
                mes = cleandoc('AWG channel {} is not currently outputting'
                               .format(self._channel))
                raise InstrIOError(mes)
            else:
                self._AWG.write('SOUR:FUNC:MODE {}'.format(value))
                if self._AWG.ask('SOUR:FUNC:MODE?') != value:
                    raise InstrIOError(cleandoc('''Instrument did not set
                                                correctly the output mode'''))
                    
    @secure_communication()
    def frequency(self):
        ''' modulation frequency getter method
        
        '''
        with self.secure():
            self._AWG.write('INST {}'.format(self._channel))
            if self.output_mode() != 'FIX':
                mes = cleandoc('AWG channel {} is not in standard mode'
                               .format(self._channel))
                raise InstrIOError(mes)
            else:
                freq = self._AWG.ask(':FREQ?')
                return float(freq)
        
    @secure_communication()
    def set_frequency(self,value):
        ''' modulation frequency setter method
        
        '''
        val = float(value)
        if not (1e3 <= val <= 1e9):
            raise InstrIOError(cleandoc('Set frequency is out of bounds'))
        with self.secure():
            self._AWG.write('INST {}'.format(self._channel))
            self._AWG.write(':FREQ {}'.format(val))
            if float(self._AWG.ask(':FREQ?')) != val:
                raise InstrIOError(cleandoc('''Instrument did not set
                                                correctly the frequency'''))    
    
    @secure_communication()
    def DC_offset(self):
        ''' DC offset getter method
        
        '''
        with self.secure():
            self._AWG.write('INST {}'.format(self._channel))
            offset = self._AWG.ask(':VOLT:OFFS?')
            return float(offset)
        
    @secure_communication()
    def set_DC_offset(self,value):
        ''' DC offset setter method
        
        '''
        val = np.round(float(value),3)
        if val > 1:
            raise InstrIOError(cleandoc('Set DC offset out of bounds'))
        with self.secure():
            self._AWG.write('INST {}'.format(self._channel))
            self._AWG.write(':VOLT:OFFS {}'.format(val))
            if float(self._AWG.ask(':VOLT:OFFS?')) != val:
                raise InstrIOError(cleandoc('''Instrument did not set
                                                correctly the DC offset'''))
    
    @instrument_property
    @secure_communication()
    def vpp(self):
        ''' Voltage peak peak getter method
        
        '''
        with self.secure():
            self._AWG.write('INST {}'.format(self._channel))
            vpp = self._AWG.ask(':VOLT?')
            return float(vpp)
        
    @vpp.setter
    @secure_communication()
    def vpp(self,value):
        ''' Voltage peak peak setter method
        
        '''
        val = np.round(float(value),3)
        if (val > 2.0) or (val < 0.05):
            raise InstrIOError(cleandoc('Set Vpp out of bounds'))
        with self.secure():
            self._AWG.write('INST {}'.format(self._channel))
            self._AWG.write(':VOLT {};*WAI'.format(val))
            if float(self._AWG.ask(':VOLT?')) != val:
                raise InstrIOError(cleandoc('''Instrument did not set
                                                correctly the Vpp'''))
    
    @secure_communication()
    def phase(self):
        ''' phase of channel getter method
        
        '''
        with self.secure():
            self._AWG.write('INST {}'.format(self._channel))
            phase = self._AWG.ask(':SIN:PHAS?')
            return float(phase)
        
    @secure_communication()
    def set_phase(self,value):
        ''' phase setter method
        
        '''
        val = np.round(float(value),3)
        if (val > 360) or (val < 0):
            raise InstrIOError(cleandoc('Set phase out of bounds'))
        with self.secure():
            self._AWG.write('INST {}'.format(self._channel))
            self._AWG.write(':SIN:PHAS {}'.format(val))
            if float(self._AWG.ask(':SIN:PHAS?')) != val:
                raise InstrIOError(cleandoc('''Instrument did not set
                                                correctly the phase'''))


class TaborAWG(VisaInstrument):
    """
    """
    caching_permissions = {'defined_channels': True}

    def __init__(self, connection_info, caching_allowed=True,
                 caching_permissions={}, auto_open=True):
        super(TaborAWG, self).__init__(connection_info, caching_allowed,
                                  caching_permissions, auto_open)
        self.channels = {}
        self.lock = Lock()
        self.write_termination = '\n'
        self.read_termination = '\n'

    def get_channel(self, num):
        """
        """
        if num not in self.defined_channels:
            return None

        if num in self.channels:
            return self.channels[num]
        else:
            channel = TaborAWGChannel(self, num)
            self.channels[num] = channel
            return channel

    @secure_communication()
    def to_send(self, waveform, ch_id):
        """Command to send to the instrument. waveform = string of a bytearray

        """
        self.write('INST {}'.format(ch_id))
        self.write('TRAC:MODE SING')
        header = "TRAC"
        self._driver.write_binary_values(header, waveform, datatype='B')

    @instrument_property
    @secure_communication()
    def defined_channels(self):
        """
        """
        channel_list = [1, 2, 3, 4]
        return channel_list

    @instrument_property
    @secure_communication()
    def oscillator_reference_external(self):
        """oscillator reference external getter method
        """
        ore = self.ask("SOUR:ROSC:SOUR?")
        if ore == 'EXT':
            return 'True'
        elif ore == 'INT':
            return 'False'
        else:
            raise InstrIOError

    @oscillator_reference_external.setter
    @secure_communication()
    def oscillator_reference_external(self, value):
        """oscillator reference external setter method
        """
        if value in ('EXT', 1, 'True'):
            self.write('SOUR:ROSC:SOUR EXT')
            if self.ask('SOUR:ROSC:SOUR?') != 'EXT':
                raise InstrIOError(cleandoc('''Instrument did not set
                                            correctly the oscillator
                                            reference'''))
        elif value in ('INT', 0, 'False'):
            self.write('SOUR:ROSC:SOUR INT')
            if self.ask('SOUR:ROSC:SOUR?') != 'INT':
                raise InstrIOError(cleandoc('''Instrument did not set
                                            correctly the oscillator
                                            reference'''))
        else:
            mess = fill(cleandoc('''The invalid value {} was sent to
                                 oscillator_reference_external
                                 method''').format(value), 80)
            raise VisaTypeError(mess)

    @instrument_property
    @secure_communication()
    def clock_source(self):
        """clock source getter method
        """

        cle = self.ask("FREQ:RAST:SOUR?")
        if cle is not None:
            return cle
        else:
            raise InstrIOError

    @clock_source.setter
    @secure_communication()
    def clock_source(self, value):
        """clock source setter method
        """
        if value in ('EXT', 1, 'True'):
            self.write(':FREQ:RAST:SOUR EXT')
            if self.ask(':FREQ:RAST:SOUR?') != 'EXT':
                raise InstrIOError(cleandoc('''Instrument did not set
                                            correctly the clock source'''))
        elif value in ('INT', 0, 'False'):
            self.write(':FREQ:RAST:SOUR INT')
            if self.ask(':FREQ:RAST:SOUR?') != 'INT':
                raise InstrIOError(cleandoc('''Instrument did not set
                                            correctly the clock source'''))
        else:
            mess = fill(cleandoc('''The invalid value {} was sent to
                                 clock_source_external
                                 method''').format(value), 80)
            raise VisaTypeError(mess)

    @instrument_property
    @secure_communication()
    def sampling_frequency(self):
        """sampling frequency getter method
        """
        sampl_freq = self.ask_for_values("FREQ:RAST?")[0]
        if sampl_freq is not None:
            return sampl_freq
        else:
            raise InstrIOError

    @sampling_frequency.setter
    @secure_communication()
    def sampling_frequency(self, value):
        """sampling frequency setter method
        """
        self.write("FREQ:RAST {}".format(value))
        result = self.ask_for_values("FREQ:RAST?")[0]
        if abs(result - value) > 10**-12:
            raise InstrIOError(cleandoc('''Instrument did not set correctly
                                        the sampling frequency'''))

    @instrument_property
    @secure_communication()
    def running(self):
        """Run state getter method
        """
        return '2 : Intrument is running'

    @running.setter
    @secure_communication()
    def running(self, value):
        """Run state setter method
        """


    @instrument_property
    @secure_communication()
    def run_mode(self):
        """Run mode getter method
        """
        run_cont = self.ask("INIT:CONT?")
        run_gat=self.ask("INIT:GATE?")
        if run_cont is not None:
            if run_cont == 'ON':
                return 'Continuous'
            elif run_gat == 'ON':
                return 'Gated'
            else:
                return 'Triggered'

        else:
            raise InstrIOError

    @run_mode.setter
    @secure_communication()
    def run_mode(self, value):
        """Run mode setter method
        """
        if value in ('CONT', 'CONTINUOUS', 'continuous'):
            self.write('INIT:CONT ON')
            if self.ask('INIT:CONT?') != 'ON':
                raise InstrIOError(cleandoc('''Instrument did not set
                                            correctly the run mode'''))
        elif value in ('TRIG', 'TRIGGERED', 'triggered'):
            self.write('INIT:CONT OFF')
            self.write('INIT:GATE OFF')
            if self.ask('INIT:CONT?') != 'OFF':
                raise InstrIOError(cleandoc('''Instrument did not set
                                            correctly the run mode'''))
            elif self.ask('INIT:GATE?') != 'OFF':
                raise InstrIOError(cleandoc('''Instrument did not set
                                            correctly the run mode'''))
        elif value in ('GAT', 'GATED', 'gated'):
            self.write('INIT:CONT OFF')
            self.write('INIT:GATE ON')
            if self.ask('INIT:CONT?') != 'OFF':
                raise InstrIOError(cleandoc('''Instrument did not set
                                            correctly the run mode'''))
            elif self.ask('INIT:GATE?') != 'ON':
                raise InstrIOError(cleandoc('''Instrument did not set
                                            correctly the run mode'''))
        else:
            mess = fill(cleandoc('''The invalid value {} was sent to
                                 run mode method''').format(value), 80)
            raise VisaTypeError(mess)
            
    @instrument_property
    @secure_communication()
    def DC_offset_range(self):
        """
        """
        DC_offset_range = (-1,1)
        return DC_offset_range

    @secure_communication()
    def set_DC_offset(self,channel,value):
        ''' sets the DC offset for a given channel'''
        channels = self.defined_channels
        if channel in channels:    
            self.write(':INST {}'.format(channel)) #select channel
            DC_min, DC_max = self.DC_offset_range
            if (value >= DC_min) and (value <= DC_max):        
                self.write(':VOLT:OFFS {}'.format(value))
            else:
                raise InstrIOError(cleandoc('''DC offset value out of range. 
                                            Allowed range: {} to {} V.'''.format(DC_min,DC_max)))
        else:
            raise InstrIOError(cleandoc('''Channel number out of range.
                                        Accepted channel numbers: {}.'''.format(channels)))