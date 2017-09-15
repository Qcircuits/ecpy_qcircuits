# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2016 by EcpyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Base classes for instrument relying on the VISA protocol.

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

from future.utils import raise_with_traceback

try:
    from pyvisa.highlevel import ResourceManager
    from pyvisa import errors
except ImportError:
    msg = 'The PyVISA library is necessary to use the visa backend.'
    raise_with_traceback(ImportError(msg))

from .driver_tools import BaseInstrument, InstrIOError


class VisaInstrument(BaseInstrument):
    """Base class for drivers using the VISA library to communicate

    This class uses the PyVisa binder to the VISA library to open a
    communication. The PyVisa object (Instrument instance) is cached and this
    class provides conveninence methods to call all its method and propeties
    to set its attributes. The connection to the instrument is opened upon
    initialisation.

    Parameters
    ----------
    connection_info : dict
        Dict containing all the necessary information to open a connection to
        the instrument
    caching_allowed : bool, optionnal
        Boolean use to determine if instrument properties can be cached
    caching_permissions : dict(str : bool), optionnal
        Dict specifying which instrument properties can be cached, override the
        default parameters specified in the class attribute.

    Attributes
    ----------
    caching_permissions : dict(str : bool)
        Dict specifying which instrument properties can be cached.
    secure_com_except : tuple(Exception)
        Tuple of the exceptions to be catched by the `secure_communication`
        decorator
    connection_str : VISA string uses to open the communication

    The following attributes simply reflects the attribute of a `PyVisa`
    `Instrument` object :
    timeout
    write_termination
    read_termination
    delay

    Methods
    -------
    open_connection() :
        Open the connection to the instrument using the `connection_str`
    close_connection() :
        Close the connection with the instrument
    reopen_connection() :
        Reopen the connection with the instrument with the same parameters as
        previously
    check_connection() : virtual
        Check whether or not the cache is likely to have been corrupted

    The following method simply call the PyVisa method of the driver
    write(mess)
    read()
    read_values()
    ask(mess)
    ask_for_values()
    clear()
    trigger()
    read_raw()

    """
    secure_com_except = (InstrIOError, errors.VisaIOError)

    def __init__(self, connection_info, caching_allowed=True,
                 caching_permissions={}, auto_open=True):
        super(VisaInstrument, self).__init__(connection_info, caching_allowed,
                                             caching_permissions)
        self.connection_str = connection_info['resource_name']

        self._driver = None
        if auto_open:
            self.open_connection()

    def open_connection(self, **para):
        """Open the connection to the instr using the `connection_str`.

        """
        rm = ResourceManager()
        try:
            self._driver = rm.open_resource(self.connection_str, **para)
        except errors.VisaIOError as er:
            self._driver = None
            raise InstrIOError(str(er))

    def close_connection(self):
        """Close the connection to the instr.

        """
        if self._driver:
            self._driver.close()
        self._driver = None
        return True

    def reopen_connection(self):
        """Reopen the connection with the instrument with the same parameters
        as previously.

        """
        para = {'timeout': self._driver.timeout,
                'query_delay': self._driver.query_delay,
                'write_termination': self._driver.write_termination,
                'read_termination': self._driver.read_termination,
                }
        self._driver.close()
        self.open_connection(**para)

    def connected(self):
        """Returns whether commands can be sent to the instrument
        """
        return bool(self._driver)

    def write(self, message):
        """Send the specified message to the instrument.

        Simply call the `write` method of the `Instrument` object stored in
        the attribute `_driver`
        """
        self._driver.write(message)

    def read(self):
        """Read one line of the instrument's buffer.

        Simply call the `read` method of the `Instrument` object stored in
        the attribute `_driver`
        """
        return self._driver.read()

    def read_values(self, format=0):
        """Read one line of the instrument's buffer and convert to values.

        Simply call the `read_values` method of the `Instrument` object
        stored in the attribute `_driver`
        """
        return self._driver.read_values(format=0)

    def ask(self, message):
        """Send the specified message to the instrument and read its answer.

        Simply call the `ask` method of the `Instrument` object stored in
        the attribute `_driver`
        """
        return self._driver.query(message)

    def ask_for_values(self, message, format=2):
        """Send the specified message to the instrument and convert its answer
        to values.

        By default assume the values are returned as ascii.

        Simply call the `ask_for_values` method of the `Instrument` object
        stored in the attribute `_driver`

        """
        return self._driver.ask_for_values(message, format)

    def clear(self):
        """Resets the device (highly bus dependent).

        Simply call the `clear` method of the `Instrument` object stored in
        the attribute `_driver`
        """
        return self._driver.clear()

    def trigger(self):
        """Send a trigger to the instrument.

        Simply call the `trigger` method of the `Instrument` object stored
        in the attribute `_driver`
        """
        return self._driver.assert_trigger()

    def read_raw(self):
        """Read one line of the instrument buffer and return without stripping
        termination caracters.

        Simply call the `read_raw` method of the `Instrument` object stored
        in the attribute `_driver`
        """
        return self._driver.read_raw()

    def _timeout(self):
        return self._driver.timeout

    def _set_timeout(self, value):
        self._driver.timeout = value

    timeout = property(_timeout, _set_timeout)
    """Conveninence to set/get the `timeout` attribute of the `Instrument`
    object"""

    def _delay(self):
        return self._driver.query_delay

    def _set_delay(self, value):
        self._driver.query_delay = value

    delay = property(_delay, _set_delay)
    """Conveninence to set/get the `query_delay` attribute of the `Instrument`
    object"""

    def _write_termination(self):
        return self._driver.write_termination

    def _set_write_termination(self, value):
        self._driver.write_termination = value

    write_termination = property(_write_termination,
                                 _set_write_termination)
    """Conveninence to set/get the `write_termination` attribute of the
    `Instrument` object"""

    def _read_termination(self):
        return self._driver.read_termination

    def _set_read_termination(self, value):
        self._driver.read_termination = value

    read_termination = property(_read_termination,
                                 _set_read_termination)
    """Conveninence to set/get the `read_termination` attribute of the
    `Instrument` object"""