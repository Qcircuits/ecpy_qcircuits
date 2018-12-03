# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2016 by EcpyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Task perform measurements with a PSA.

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

import numpy as np
from atom.api import set_default, Unicode

from ecpy.tasks.api import InstrumentTask

class PSAGetSpectrumTask(InstrumentTask):
    """ Get the trace that is displayed right now

    """    
    database_entries = set_default({'sweep_data': {}})

    def perform(self):
        if not self.driver:
            self.start_driver()
            
        tr_data = self.get_trace()
        self.write_in_database('sweep_data',tr_data)
        
    def get_trace(self):
        """ Get the trace that is displayed right now (no new acquisition)

        """
        data = self.driver.read_raw_data((1,1))
        data_x = self.driver.get_x_data((1,1))
        aux = [data_x,data]
        return np.rec.fromarrays(aux,names=['Freq (GHz)','Spectrum'])
    
class PSAGetFrequencyPointTask(InstrumentTask):
    """ Get the power of a single frequency point extracted from the spectrum

    """    
    frequency = Unicode('1').tag(pref=True)
    power_level = Unicode('0').tag(pref=True)
    
    database_entries = set_default({'power': {}})
    
    def perform(self):
        if not self.driver:
            self.start_driver()
            
        data = self.get_point()
        self.write_in_database('power',data)
        
    def get_point(self):
        """ Get the frequency point given

        """
        data = self.driver.get_single_freq(\
                                           self.format_and_eval_string(
                                                   self.frequency)*1e9,
                                           self.format_and_eval_string(
                                                   self.power_level),
                                                   1000,1000,10)
        aux = [data]
        return np.rec.fromarrays(aux,names=['Power'])