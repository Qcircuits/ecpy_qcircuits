# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2022 by Qcircuit Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Driver for the Keysight programmable DC power supplies

"""
import traceback
import threading
import re
from textwrap import fill
from inspect import cleandoc
import time
from pyvisa import constants

from exopy_hqc_legacy.instruments.drivers.driver_tools import (
    InstrIOError, instrument_property, secure_communication)
from exopy_hqc_legacy.instruments.drivers.visa_tools import VisaInstrument, errors


class E3631A(VisaInstrument):
    """
    Driver for the E3631A, using the VISA library.

    This driver does not give access to all the functionnality of the
    instrument but you can extend it if needed. See the documentation of the
    `driver_tools` package for more details about writing instruments drivers.

    Parameters
    ----------
    see the `VisaInstrument` parameters in the `driver_tools` module

    Attributes
    ----------
    channel: str, instrument_property
        Currently selected channel. Can be 'P6V', 'P25V' or 'N25V'
    voltage : float, instrument_property
        Voltage at the output of the current channel in volts.
    current: float, instrument_property
        Current at the output of the channel in amps.
    output : bool, instrument_property
        State of the output 'ON'(True)/'OFF'(False).

    """

    def open_connection(self, **para):
        """Open the connection to the instr

        """
        super(E3631A, self).open_connection(**para)
        self.timeout = 3000

    @instrument_property
    def channel(self):
        """Channel getter method.

        """
        c = self.query('INST:SEL?')
        if c:
            return c[:-1]
        raise InstrIOError('Instrument did not return the channel')

    @channel.setter
    def channel(self, target_channel):
        """Channel getter method.

        """
        if target_channel not in ['P6V', 'P25V', 'N25V']:
            raise InstrIOError(f'Invalid channel name {target_channel}')
        self.write(f'INST:SEL {target_channel}')
        c = self.query('INST:SEL?')
        if not (c and c[:-1] == target_channel):
            raise InstrIOError("Instrument couldn't set the channel")

    @instrument_property
    def voltage(self):
        """Voltage getter method.

        """
        v = self.query('VOLT?')
        if v:
            return float(v)
        else:
            raise InstrIOError('Instrument did not return the voltage')

    @voltage.setter
    def voltage(self, set_point):
        """Voltage setter method.

        """
        self.write(f"VOLT {set_point}")
        v = self.query('VOLT?')
        if not v or abs(float(v) - set_point) > 10**-3:
            raise InstrIOError('Instrument did not set correctly the voltage')

    @instrument_property
    def current(self):
        """Current getter method.

        """
        v = self.query('CURR?')
        if v:
            return float(v)
        else:
            raise InstrIOError('Instrument did not return the current')

    @current.setter
    def current(self, set_point):
        """Current setter method.

        """
        self.write(f"CURR {set_point}")
        v = self.query('CURR?')
        if not v or abs(float(v) - set_point) > 10**-3:
            raise InstrIOError('Instrument did not set correctly the current')

    @instrument_property
    def output(self):
        """Output getter method

        """
        v = self.query('OUTP:STAT?')
        if v:
            return '1' in v
        raise InstrIOError('Instrument did not return the output')

    @output.setter
    def output(self, value):
        """Output setter method

        """
        on = re.compile('on', re.IGNORECASE)
        off = re.compile('off', re.IGNORECASE)
        if on.match(value) or value == 1:
            self.write('OUTP:STAT ON')
            time.sleep(0.1)
            v = self.query('OUTP:STAT?')
            status = '1' in v
            if not status:
                raise InstrIOError(
                    cleandoc('''Instrument did not set
                                            correctly the output'''))
        elif off.match(value) or value == 0:
            self.write('OUTP:STAT OFF')
            time.sleep(0.1)
            v = self.query('OUTP:STAT?')
            status = '1' in v
            if status:
                raise InstrIOError(
                    cleandoc('''Instrument did not set
                                            correctly the output'''))

    def check_connection(self):
        """
        """
        return False