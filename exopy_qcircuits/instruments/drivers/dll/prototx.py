# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2022 by exopyQcircuitsLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""

This module defines drivers for ProtoTx using DLL Library.

"""
from lib2to3.pgen2.token import OP
from ..driver_tools import (InstrError, InstrIOError, secure_communication,
                            instrument_property)
from ..dll_tools import DllLibrary, DllInstrument
from enum import Enum
from inspect import cleandoc
import ctypes


class ProtoTxDll(DllLibrary):
    """ Wrapper for the ProtoTx dll library (protoTx.dll).

    """
        
    def __init__(self, path, **kwargs):

        super(ProtoTxDll, self).__init__(path, **kwargs)

        # See what we have connected
        self.handles = self.connected_instruments()

        # Initialize all the settings
        self.I_offset = None
        self.Q_offset = None
        self.f_LO = None
        self.max_power = None
        self.RF_power = None
        self.RF_attenuation = None
        self.operating_mode = None
        self.preset_mode = None
        self.ref_source = None
        self.LO_nulling = None
        self.RF_filter = None
        self.RF_output = None
        self.LO_source = None
        self.min_power = None


    def connected_instruments(self):
        """ Return the handles of all connected instruments.

        """
        #print("Enumerating devices")
        MAX_DEVICES = 5
        status = ctypes.c_ulong()
        devices_found = ctypes.c_int()
        handles = (MAX_DEVICES * ctypes.c_void_p)()
        serial_numbers = (MAX_DEVICES * ctypes.c_int)()
        r = self.dll.protoTx_Open(ctypes.byref(status), ctypes.byref(devices_found), 
                                  ctypes.byref(handles), ctypes.byref(serial_numbers))
        #print(devices_found.value)
        #print(handles[0])
        #print(serial_numbers[0])
        if r == 0:
            return handles
        elif r == -1:
            raise InstrIOError("Couldn't find any protoTx to connect to")
        else:
            raise InstrIOError(f"FTDI error: {status.value}")

    def disconnect(self, handle):
        """
        Close the connection to all instruments
        """
        #print(f"Closing connection to handle {handle}")
        self.dll.protoTx_Close(ctypes.c_void_p(handle))


# Some properties of the device
    def read_status(self, handle):
        #print("Reading status")
        status = ctypes.c_ulong()
        values = (14*ctypes.c_double)()

        #print(f"Using handle {handle}")

        self.dll.protoTx_ReadStatus(ctypes.byref(status), ctypes.c_void_p(handle), ctypes.byref(values))

        #for x in values:
        #    print(x)  
        self.I_offset = values[0]
        self.Q_offset = values[1]
        self.f_LO = values[2]*1e-3
        self.max_power = values[3]
        self.RF_power = values[4]
        self.RF_attenuation = values[5]
        self.operating_mode = ProtoTx.OperatingMode(int(values[6]))
        self.preset_mode = values[7]
        self.ref_source = ProtoTx.Source(int(values[8]))
        self.LO_nulling = ProtoTx.LOnulling(int(values[9]))
        self.RF_filter = ProtoTx.RFFilter(int(values[10]))
        self.RF_output = values[11]
        self.LO_source = ProtoTx.Source(int(values[12]))
        self.min_power = values[13]

    def write_status(self, handle):
        #print("Writing status")
        #print(f"Using handle {handle}")
 
        status = ctypes.c_ulong()
        values = (12*ctypes.c_double)()

        values[0] = float(self.I_offset)
        values[1] = float(self.Q_offset)
        values[2] = float(self.f_LO * 1e3)
        values[3] = float(self.RF_power)
        values[4] = float(self.RF_attenuation)
        values[5] = float(self.operating_mode.value)
        values[6] = float(self.preset_mode)
        values[7] = float(self.ref_source.value)
        values[8] = float(self.LO_nulling.value)
        values[9] = float(self.RF_filter.value)
        values[10] = float(self.RF_output)
        values[11] = float(self.LO_source.value)

        #for x in values:
        #    print(x) 
        self.dll.protoTx_WriteStatus(ctypes.byref(status), ctypes.c_void_p(handle), ctypes.byref(values))


class ProtoTx(DllInstrument):
    class OperatingMode(Enum):
        IQ = 0
        UPPER = 1
        LOWER = 2
        SYNTHETIZER = 3

    class Source(Enum):
        INTERNAL = 0
        EXTERNAL = 1

    class LOnulling(Enum):
        MANUAL = 0
        FACTORY = 2

    class RFFilter(Enum):
        f_500MHz = 0
        f_950MHz = 1
        f_1700MHz = 2
        f_3250MHz = 3
        f_5500MHz = 4


    library = 'protoTx_patched.dll'

    def __init__(self, connection_info, caching_allowed=True,
                 caching_permissions={}, auto_open=True):

        library = 'protoTx_patched.dll'


        super(ProtoTx, self).__init__(connection_info, caching_allowed,
                                             caching_permissions, auto_open)

        self._dll = ProtoTxDll(connection_info['lib_dir'] + '/'+library)

        self.open_connection()
        self._dll.read_status(self._dll.handles[0])

        # self.frequency = 4
        # self.operating_mode = OperatingMode.IQ
        # self.output = 0
        # self.I_offset = 127
        # self.Q_offset = 127
        # self.rf_attenuation = 0
        # self.reference_source = Source.EXTERNAL
        # self.lo_source = Source.INTERNAL

    def open_connection(self):
        """ Open a connection to the instrument.

        """
        self.connected = True

    def close_connection(self):
        """ Close the connection established previously using `open_connection`

        """
        self._dll.disconnect(self._dll.handles[0])
        self.connected = False

    def reopen_connection(self):
        """ Reopen connection established previously using `open_connection`

        """
        self._dll.connected_instruments()
        self.connected = True

    def connected(self):
        """ Check whether or not the instrument is connected

        """
        return self.connected

    @instrument_property
    @secure_communication()
    def max_power(self):
        return self._dll.max_power

    @instrument_property
    @secure_communication()
    def min_power(self):
        return self._dll.min_power

    @instrument_property
    @secure_communication()
    def I_offset(self):
        return self._dll.I_offset

    @I_offset.setter
    @secure_communication()
    def I_offset(self, value):
        """ I offset setter method.
        """
        if not (0<=value<=255):
            mes = 'I offset must be between 0 and 255'
            raise ValueError(mes)
        if value - int(value) > 10**-5:
            mes = 'I offset must be a float with an integer value'
            raise ValueError(mes)
        self._dll.LO_nulling = ProtoTx.LOnulling(0)
        self._dll.I_offset = value
        self._dll.write_status(self._dll.handles[0])
        self._dll.read_status(self._dll.handles[0])

        if abs(self._dll.I_offset - value) > 10**-5:
            mes = 'Instrument did not set correctly the I offset'
            raise InstrIOError(mes)


    @instrument_property
    @secure_communication()
    def Q_offset(self):
        return self._dll.Q_offset

    @Q_offset.setter
    @secure_communication()
    def Q_offset(self, value):
        """ Q offset setter method.
        """
        if not (0<=value<=255):
            mes = 'I offset must be between 0 and 255'
            raise ValueError(mes)
        if value - int(value) > 10**-5:
            mes = 'I offset must be a float with an integer value'
            raise ValueError(mes)
        self._dll.LO_nulling = ProtoTx.LOnulling(0)
        self._dll.Q_offset = value
        self._dll.write_status(self._dll.handles[0])
        self._dll.read_status(self._dll.handles[0])

        if abs(self._dll.Q_offset - value) > 10**-5:
            mes = 'Instrument did not set correctly the Q offset'
            raise InstrIOError(mes)        

    @instrument_property
    @secure_communication()
    def frequency(self):
        ''' frequency getter method.

        '''
        return self._dll.f_LO

    @frequency.setter
    @secure_communication()
    def frequency(self, value):
        """ Frequency setter method.

        Input : frequency in GHz

        """
        self._dll.f_LO = value
        self._dll.write_status(self._dll.handles[0])
        self._dll.read_status(self._dll.handles[0])

        if abs(self._dll.f_LO - value) > 10**-9:
            mes = 'Instrument did not set correctly the frequency'
            raise InstrIOError(mes)

        if value < 0.2:
            self.rf_filter = ProtoTx.RFFilter.f_500MHz
        elif value < 0.65:
            self.rf_filter = ProtoTx.RFFilter.f_950MHz
        elif value < 1.4:
            self.rf_filter = ProtoTx.RFFilter.f_1700MHz
        elif value < 2.95:
            self.rf_filter = ProtoTx.RFFilter.f_3250MHz
        else:
            self.rf_filter = ProtoTx.RFFilter.f_5500MHz

    @instrument_property
    @secure_communication()
    def power(self):
        ''' power getter method.

        '''
        return self._dll.RF_power

    @power.setter
    @secure_communication()
    def power(self, value):
        ''' power setter method.
        Input : number, No string

        '''
        if self.operating_mode != ProtoTx.OperatingMode.SYNTHETIZER:
            mes = "Can only set the power in synthetizer mode"
            raise InstrIOError(mes)
        if value%0.25 > 10**-5:
            mes = "Power must be multiple of 0.25"
            raise ValueError(mes)

        self._dll.RF_power = value
        self._dll.write_status(self._dll.handles[0])
        self._dll.read_status(self._dll.handles[0])

        if abs(self._dll.f_LO - value) > 0.1:
            mes = 'Instrument did not set correctly the power'
            raise InstrIOError(mes)

    @instrument_property
    @secure_communication()
    def lo_source(self):
        ''' LO source getter

        '''
        return self._dll.LO_source

    @lo_source.setter
    @secure_communication()
    def lo_source(self, value):
        ''' LO source setter

        '''

        self._dll.LO_source = value
        self._dll.write_status(self._dll.handles[0])
        self._dll.read_status(self._dll.handles[0])

        if self._dll.LO_source != value:
            mes = 'Instrument did not set correctly the LO source'
            raise InstrIOError(mes)

    @instrument_property
    @secure_communication()
    def output(self):
        ''' Output getter method.

        '''
        return self._dll.RF_output

    @output.setter
    @secure_communication()
    def output(self, value):
        ''' Output setter method.
        Input = {1 = on, 0 = off} (careful: No string)
        
        '''
        tmp = value
        if value == 'On':
            tmp = 1.0
        elif value == "Off":
            tmp = 0.0
        self._dll.RF_output = tmp
        self._dll.write_status(self._dll.handles[0])
        self._dll.read_status(self._dll.handles[0])

        if self._dll.RF_output != tmp:
            mes = 'Instrument did not set correctly the RF output'
            raise InstrIOError(mes)

    @instrument_property
    @secure_communication()
    def rf_attenuation(self):
        ''' RF attenuation getter method.
        Input = None
        Output = Attenuation in dB

        '''
        return self._dll.RF_attenuation

    @rf_attenuation.setter
    @secure_communication()
    def rf_attenuation(self, value):
        ''' RF attenuation setter method.
        Input = Attenuation in dB from 0 to 31.75 in steps of 0.25
        Output = None

        '''
        if self.operating_mode == ProtoTx.OperatingMode.SYNTHETIZER:
            mes = "Cannot set the attenuation in synthetizer mode"
            raise InstrError(mes)
        if value%0.25 > 10**-5:
            mes = "Attenuation must be multiple of 0.25"
            raise ValueError(mes)
        self._dll.RF_attenuation = value
        self._dll.write_status(self._dll.handles[0])
        self._dll.read_status(self._dll.handles[0])

        if self._dll.RF_attenuation != value:
            mes = 'Instrument did not set correctly the RF attenuation'
            raise InstrIOError(mes)

    @instrument_property
    @secure_communication()
    def operating_mode(self):
        ''' Operating mode getter method.

        '''
        return self._dll.operating_mode

    @operating_mode.setter
    @secure_communication()
    def operating_mode(self, value):
        ''' Operating mode setter method.

        '''
        self._dll.operating_mode = value
        self._dll.write_status(self._dll.handles[0])
        self._dll.read_status(self._dll.handles[0])

        if self._dll.operating_mode != value:
            mes = 'Instrument did not set correctly the operating mode'
            raise InstrIOError(mes)

    @instrument_property
    @secure_communication()
    def reference_source(self):
        ''' Reference source getter method.

        '''
        return self._dll.ref_source

    @reference_source.setter
    @secure_communication()
    def reference_source(self, value):
        ''' Reference setter method.

        '''
        self._dll.ref_source = value
        self._dll.write_status(self._dll.handles[0])
        self._dll.read_status(self._dll.handles[0])

        if self._dll.ref_source != value:
            mes = 'Instrument did not set correctly the operating mode'
            raise InstrIOError(mes)
            
    @instrument_property
    @secure_communication()
    def rf_filter(self):
        ''' RF output filter getter method.

        '''
        return self._dll.RF_filter

    @rf_filter.setter
    @secure_communication()
    def rf_filter(self, value):
        ''' RF output filter setter method.

        '''
        self._dll.RF_filter = value
        self._dll.write_status(self._dll.handles[0])
        self._dll.read_status(self._dll.handles[0])

        if self._dll.RF_filter != value:
            mes = 'Instrument did not set correctly the RF output filter'
            raise InstrIOError(mes) 

    @instrument_property
    @secure_communication()
    def lo_nulling(self):
        ''' LO nulling getter method.

        '''
        return self._dll.LO_nulling

    @lo_nulling.setter
    @secure_communication()
    def lo_nulling(self, value):
        ''' LO nulling setter method.

        '''
        if not (value == ProtoTx.LOnulling.FACTORY):
            mes = "can only set LO nulling to factory mode, LO nulling is automatically switched to manual with a change of I/Q offsets"
            raise InstrError(mes)
        self._dll.LO_nulling = value
        self._dll.write_status(self._dll.handles[0])
        self._dll.read_status(self._dll.handles[0])

        if self._dll.LO_nulling != value:
            mes = 'Instrument did not set correctly the LO nulling'
            raise InstrIOError(mes)             


    @secure_communication()
    def set_all_config(self, I_offset, Q_offset, operating_mode, ref_source, LO_nulling, LO_source):
        self._dll.I_offset = I_offset
        self._dll.Q_offset = Q_offset
        self._dll.operating_mode = operating_mode
        self._dll.ref_source = ref_source
        self._dll.LO_nulling = LO_nulling
        self._dll.LO_source = LO_source
        self._dll.write_status(self._dll.handles[0])
        self._dll.read_status(self._dll.handles[0])

        if self._dll.LO_nulling != LO_nulling:
            mes = 'Instrument did not set correctly the LO nulling'
            raise InstrIOError(mes)     
        if self._dll.I_offset != I_offset:
            mes = 'Instrument did not set correctly the I offset'
            raise InstrIOError(mes)
        if self._dll.Q_offset != Q_offset:
            mes = 'Instrument did not set correctly the Q offset'
            raise InstrIOError(mes)
        if self._dll.operating_mode != operating_mode:
            mes = 'Instrument did not set correctly the operating mode'
            raise InstrIOError(mes)
        if self._dll.ref_source != ref_source:
            mes = 'Instrument did not set correctly the reference source'
            raise InstrIOError(mes)
        if self._dll.LO_source != LO_source:
            mes = 'Instrument did not set correctly the LO source'
            raise InstrIOError(mes)