#-*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by exopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Base classes for instrument relying on a custom dll for communication.

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

import time
import zhinst.utils
import sys, os


from .driver_tools import BaseInstrument, InstrIOError


class ZIInstrument(BaseInstrument):
    """ A base class for all Zurich Instrument instrumensts.

    Attributes
    ----------

    """
    
    def __init__(self, connection_info, caching_allowed=True,
                 caching_permissions={}, auto_open=True):
        super(ZIInstrument, self).__init__(connection_info, caching_allowed,
                                             caching_permissions)

        self.apilevel=6
        self.device_number=connection_info['device_number']
        self.required_devtype='.*'
        self.required_options=None
        self.required_err_msg=''
        
        self.saveConfig = False
        
        self._driver = None

        
        self.daq = None
        self.device= None
        self.props = None
        if auto_open:
            self.open_connection()

    def open_connection(self):
        """Open the connection to the instr using the `connection_str`.

        """

        try:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            (self.daq,self.device,self.props) = zhinst.utils.create_api_session(self.device_number, 
                                                                                self.apilevel,self.required_devtype,
                                                                                self.required_options, self.required_err_msg);
            self._driver = 1
            sys.stdout = old_stdout
        except RuntimeError  as er:
            self.daq = None
            self.device= None
            self.props = None
            self._driver = None
            raise InstrIOError(str(er))
        except Exception  as ex:
            self.daq = None
            self.device= None
            self.props = None
            self._driver = None
            raise InstrIOError(str(ex))
        
        if self.daq and not zhinst.utils.api_server_version_check(self.daq):
            raise InstrIOError('VersionError: the release version of the API used in the session (daq) does not have the same release version as the Data Server')
        if self.daq:
            self._driver=1

       # if self.props:
       #     print(self.props['connected'])
    def close_connection(self):
        """Close the connection to the instr.

        """
        if self._driver:
            self.daq.disconnect()
        self.daq = None
        self.device= None
        self.props = None
        self._driver = None
        return True

    def reopen_connection(self):
        """Reopen the connection with the instrument with the same parameters
        as previously.

        """
        self.close()
        self.open_connection()

    def connected(self):
        """Returns whether commands can be sent to the instrument
        """
        return bool(self._driver)

    def clear(self):
        """Resets the device (highly bus dependent).

        Simply call the `clear` method of the `Instrument` object stored in
        the attribute `_driver`
        """
        return self.daq.disconnect()

    def saveConfiguration(self,full_path):
        """ Save the configuration of the LabOne interface.
        
        Give a xml file (can be load in the LabOne intertace) saved in the folder 
        given by full_path
        """

        zhinst.utils.save_settings(self.daq,self.device,full_path)            
        self.saveConfig = True
