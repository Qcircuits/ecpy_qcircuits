# -*- coding: utf-8 -*-
"""This module defines drivers for UHFLI using Zhinst Library.

:Contains:
    UHFLI

Python package zhinst from Zurich Instruments needs to be installed

"""

from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

import sys
from subprocess import call
import ctypes
import os
from inspect import cleandoc
import math

import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

try:
    from ..ZI_tools import ZIInstrument

    import zhinst.utils

    class HDAWG(ZIInstrument):
        
        def __init__(self,connection_info, caching_allowed=True,
                    caching_permissions={}):
            super(HDAWG, self).__init__(connection_info, caching_allowed,
                                                caching_permissions)
            self.awgModule = None
            self.required_devtype='.*HDAWG'
            self.required_options=['AWG','DIG']
            self.channels = [0,1,2,3,4,5,6,7]
            
        def close_connection(self):
            
            if self.awgModule:
                if self.daq:
                    self.daq.setInt('/%s/awgs/0/enable' %self.device, 0)
                    self.awgModule.finish()
                    self.awgModule.clear()
                
        def set_general_setting(self):
            general_setting = [['/%s/demods/*/enable' % self.device, 0],
                        ['/%s/scopes/*/enable' % self.device, 0]]
            self.daq.set(general_setting)
            self.daq.sync()
            
        def TransferSequence(self,awg_program):
            # Transfer the AWG sequence program. Compilation starts automatically.
            self.awgModule.set('awgModule/compiler/sourcestring', awg_program)
            
            while self.awgModule.getInt('awgModule/compiler/status') == -1:
                time.sleep(0.1)

            if self.awgModule.getInt('awgModule/compiler/status') == 1:
            # compilation failed, raise an exception
                raise Exception(self.awgModule.getString('awgModule/compiler/statusstring'))
            else:
                if self.awgModule.getInt('awgModule/compiler/status') == 2:
                    print("Compilation successful with warnings, will upload the program to the instrument.")
                    print("Compiler warning: ",
                        self.awgModule.getString('awgModule/compiler/statusstring'))
            
            # Wait for the waveform upload to finish
            time.sleep(0.2)
            i = 0
            while (self.awgModule.getDouble('awgModule/progress') < 1.0) and (self.awgModule.getInt('awgModule/elf/status') != 1):
                print("{} awgModule/progress: {:.2f}".format(i, self.awgModule.getDouble('awgModule/progress')))
                time.sleep(0.2)
                i += 1
            print("{} awgModule/progress: {:.2f}".format(i, self.awgModule.getDouble('awgModule/progress')))
            if self.awgModule.getInt('awgModule/elf/status') == 0:
                print("Upload to the instrument successful.")
            if self.awgModule.getInt('awgModule/elf/status') == 1:
                raise Exception("Upload to the instrument failed.")


except ModuleNotFoundError:
    logger.info("Couldn't find the zhinst module, please install the driver "
                "if you want to use the HDAWG.")

    # Stub class to remove make exopy happy
    class HDAWG:
        pass
