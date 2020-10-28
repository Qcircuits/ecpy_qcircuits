# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2018 by ExopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Driver for the TENMA programmable DC power supplies

"""
import re
from textwrap import fill
from inspect import cleandoc
import time
from pyvisa import constants

from exopy_hqc_legacy.instruments.drivers.driver_tools import (InstrIOError, instrument_property,
                            secure_communication)
from exopy_hqc_legacy.instruments.drivers.visa_tools import VisaInstrument, errors


class Tenma(VisaInstrument):
    """
    Driver for the Tenma, using the VISA library.

    This driver does not give access to all the functionnality of the
    instrument but you can extend it if needed. See the documentation of the
    `driver_tools` package for more details about writing instruments drivers.

    Parameters
    ----------
    see the `VisaInstrument` parameters in the `driver_tools` module

    Attributes
    ----------
    voltage : float, instrument_property
        Voltage at the output of the generator in volts.
    current: float, instrument_property
        Current at the output of the generator in amps.
    output : bool, instrument_property
        State of the output 'ON'(True)/'OFF'(False).

    """
    def open_connection(self, **para):
        """Open the connection to the instr using the `connection_str`.

        """
        para['baud_rate'] = 9600
        # Horrible hack required to make the connection work on my PC
        attempts = 0
        while attempts < 100000:
            try:
                attempts += 1
                super(Tenma, self).open_connection(**para)
                break
            except:
                continue
        self.timeout = 1000
        self.write_termination = ''
        self.read_termination = ''
        self.write('*CLS')
        time.sleep(0.1)

    @instrument_property
    @secure_communication()
    def voltage(self):
        """Voltage getter method.

        """
        self.write('VSET1?')        
        voltage = self.read_bytes(5)
        if voltage:
            return float(voltage)
        else:
            raise InstrIOError('Instrument did not return the voltage')

    @voltage.setter
    @secure_communication()
    def voltage(self, set_point):
        """Voltage setter method.

        """
        time.sleep(0.1)
        self.write("VSET1:{}".format(set_point))
        time.sleep(0.1)
        self.write('VSET1?')
        value = self.read_bytes(5)
        if abs(float(value) - set_point) > 10**-3:
            raise InstrIOError('Instrument did not set correctly the voltage')

    @instrument_property
    @secure_communication()
    def current(self):
        """Current getter method.

        """
        self.write('ISET1?')
        value = self.read_bytes(6)
        if value:
            return float(value)
        else:
            raise InstrIOError('Instrument did not return the current')

    @current.setter
    @secure_communication()
    def current(self, set_point):
        """Current setter method.

        """
        time.sleep(0.1)
        self.write("ISET1:{}".format(set_point))
        time.sleep(0.1)
        self.write('ISET1?')
        value = self.read_bytes(6)
        if abs(float(value) - set_point) > 10**-3:
            raise InstrIOError('Instrument did not set correctly the current')

    @instrument_property
    @secure_communication()
    def output(self):
        """Output getter method

        """
        self.write('STATUS?')
        value = self.read_bytes(1)
        if value:
            return bool(int.from_bytes(value, 'big') & 0b0100_0000)
        else:
            raise InstrIOError('Instrument did not return the output state')

    @output.setter
    @secure_communication()
    def output(self, value):
        """Output setter method

        """
        on = re.compile('on', re.IGNORECASE)
        off = re.compile('off', re.IGNORECASE)
        if on.match(value) or value == 1:
            self.write('OUT1')
            time.sleep(0.1)
            self.write('STATUS?')
            status = int.from_bytes(self.read_bytes(1), 'big')
            if not status & 0b0100_0000 :
                raise InstrIOError(cleandoc('''Instrument did not set
                                            correctly the output'''))
        elif off.match(value) or value == 0:
            self.write('OUT0')
            time.sleep(0.1)
            self.write('STATUS?')
            status = int.from_bytes(self.read_bytes(1), 'big')
            if status & 0b0100_0000 :
                raise InstrIOError(cleandoc('''Instrument did not set
                                            correctly the output'''))

    def check_connection(self):
        """
        """
        return False
