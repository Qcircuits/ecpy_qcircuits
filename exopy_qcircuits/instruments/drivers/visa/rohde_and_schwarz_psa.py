# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2016 by exopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Driver for the Keysight VNA (PNA).

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

from inspect import cleandoc
import numpy as np

try:
    from pyvisa import ascii, single, double
except ImportError:
    ascii = 2
    single = 1
    double = 3

from ..driver_tools import (InstrIOError, secure_communication,
                            instrument_property)
from ..visa_tools import VisaInstrument


FORMATTING_DICT = {'PHAS': lambda x: np.angle(x, deg=True),
                   'MLIN': np.abs,
                   'MLOG': lambda x: 10*np.log10(np.abs(x)),
                   'REAL': np.real,
                   'IMAG': np.imag}


class RohdeAndSchwarzPSA(VisaInstrument):
    """
    """
    _channel = 1
    port = 1
    caching_permissions = {'frequency': True,
                           'power': True,
                           'selected_measure': True,
                           'if_bandwidth': True,
                           'sweep_type': True,
                           'sweep_points': True,
                           'average_state': True,
                           'average_count': True,
                           'average_mode': True}

    def __init__(self, connection_info,caching_allowed=True,
                 caching_permissions={},auto_open=True):
        super(RohdeAndSchwarzPSA, self).__init__(connection_info, caching_allowed,
                                                caching_permissions,auto_open=True)
        
    def _close(self):
        self.close_connection()


#    def reopen_connection(self):
#        """
#        """
#        self._pna.reopen_connection()


    @secure_communication()
    def read_raw_data(self, measname):
        """ Read raw data for a measure.

        Parameters
        ----------
        channel : int
        trace : int

        Returns
        -------
        data : numpy.array
            Array of Floating points holding the data.

        """
        #stop continuous measure mode
        self.write('INIT:CONT OFF')
        #start new measurement and wait before continuing with the commands
        self.write('INIT;*WAI')
        #get sweep data
        trace = measname[1]
        data_request = 'TRAC? TRACE{}'.format(trace)
        data = np.array(self.ask(data_request).split(','),dtype = float)
        
        if list(data):
            return data
        else:
            raise InstrIOError(cleandoc('''Rhode&Schwarz PSA did not return the
                                        data for trace {}'''.format(trace)))
    
    @secure_communication()
    def get_x_data(self, measname):
        """ Read raw data for a measure.

        Parameters
        ----------
        channel : int
        trace : int

        Returns
        -------
        data : numpy.array
            Array of Floating points holding the data.

        """
        trace = measname[1]
        data_request = 'TRAC:X? TRACE{}'.format(trace)
        data = np.array(self.ask(data_request).split(','),dtype = float)
        

        if list(data):
            return data
        else:
            raise InstrIOError(cleandoc('''Rhode&Schwarz PSA did not return the
                                        x values for trace {}'''.format(trace)))
    
    @instrument_property
    @secure_communication()
    def resBW(self):
        rbw = self.ask(':BAND?')
        return rbw
    
    @secure_communication()
    @resBW.setter
    def resBW(self,value):
        self.write(':BAND {}'.format(value))
        if self.ask(':BAND?') != value:
            raise InstrIOError(cleandoc('''Rhode&Schwarz PSA did not set the 
                                        res BW properly'''))
    
    @instrument_property
    @secure_communication()
    def videoBW(self):
        vbw = self.ask(':BAND:VID?')
        return vbw
    
    @secure_communication()
    @videoBW.setter
    def videoBW(self,value):
        self.write(':BAND:VID {}'.format(value))
        if self.ask(':BAND:VID?') != value:
            raise InstrIOError(cleandoc('''Rhode&Schwarz PSA did not set the 
                                        video BW properly'''))
            
    @instrument_property
    @secure_communication()
    def span(self):
        span = self.ask(':FREQ:SPAN?')
        return span
    
    @secure_communication()
    @span.setter
    def span(self,value):
        self.write(':FREQ:SPAN {}'.format(value))
        if self.ask(':FREQ:SPAN?') != value:
            raise InstrIOError(cleandoc('''Rhode&Schwarz PSA did not set the 
                                        frequency span properly'''))
    
    @secure_communication()
    def get_single_freq(self,freq,reflevel,rbw,vbw,avrg_num):
        # get the single frequencies with sensible PSA parameters
        values = []
        for i in range(avrg_num):
            values.append(float(self.ask(
                    'SENSe:LIST:POW? {},{},0,OFF,NORM,{},{},500us,0;*OPC'.
                                 format(freq,reflevel,rbw,vbw))))
        
        return 10*np.log10(np.average(10**(np.array(values)/10)))       