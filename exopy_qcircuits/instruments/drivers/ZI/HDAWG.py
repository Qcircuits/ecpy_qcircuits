# -*- coding: utf-8 -*-
"""This module defines drivers for UHFLI using Zhinst Library.

:Contains:
    UHFLI

Python package zhinst from Zurick Instruments need to be install 

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
        
    def TransfertSequence(self,awg_program):
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
        # wait for waveform upload to finish
        i = 0
        while self.awgModule.getDouble('awgModule/progress') < 1.0:
            time.sleep(0.1)
            i += 1
        
    def get_DAQmodule(self, DAM, dimensions, signalID,signal_paths):
        print(signalID)
        data={i:[] for i in signalID}
        # Start recording data.
        DAM.set('dataAcquisitionModule/endless', 0);
        t0 = time.time()
        # Record data in a loop with timeout.
        timeout =dimensions[0]*dimensions[1]*dimensions[2]*0.001+10
        DAM.execute()
        #while not DAM.finished():
        while not DAM.finished():
            if time.time() - t0 > timeout:
                raise Exception("Timeout after {} s - recording not complete.".format(timeout))
            
            data_read = DAM.read(True)
            for sp,sid in np.transpose([signal_paths,signalID]):
                if sp in data_read.keys():
                    for d in data_read[sp]:
                        if d['header']['flags'] & 1:
                            data[sid].append(d['value'])
        DAM.finish()
        # There may be new data between the last read() and calling finished().
        data_read = DAM.read(True)
        for sp,sid in np.transpose([signal_paths,signalID]):
                if sp in data_read.keys():
                    for d in data_read[sp]:
                        if d['header']['flags'] & 1:
                            data[sid].append(d['value'])
        DAM.clear()
    
        answerTypeGrid=[]
        for sid  in signalID:
            answerTypeGrid = answerTypeGrid+ [(sid,str(data[sid][0][0][0].dtype))]
              

        answerDAM = np.zeros((dimensions[0],dimensions[1],dimensions[2]), dtype=answerTypeGrid)
        
        for sid in signalID:
            answerDAM[sid] = data[sid]
        return answerDAM

