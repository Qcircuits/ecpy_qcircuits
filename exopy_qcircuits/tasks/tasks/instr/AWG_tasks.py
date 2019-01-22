# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2016 by exopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Sets AWG parameters.

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

#import numpy as np
#from atom.api import set_default

from exopy.tasks.api import InstrumentTask
from atom.api import Unicode

class AWGSetDCOffsetTask(InstrumentTask):
    """ Set the DC offset voltage of a given AWG channel

    """    
    
    channel = Unicode('1').tag(pref=True)
    voltage = Unicode('0').tag(pref=True)

    def perform(self):
        """
        """
        if not self.driver:
            self.start_driver()
        
        channel_driver = self.driver.get_channel(int(self.channel))
        channel_driver.set_DC_offset( \
                        float(self.format_and_eval_string(self.voltage)))
        
class AWGSetVppTask(InstrumentTask):
    """ Set the Vpp of a given AWG channel

    """    
    
    channel = Unicode('1').tag(pref=True)
    amplitude = Unicode('0').tag(pref=True)

    def perform(self):
        """
        """
        if not self.driver:
            self.start_driver()
        
        channel_driver = self.driver.get_channel(int(self.channel))
        #channel_driver.set_Vpp(float(self.format_and_eval_string(self.
                                                                #amplitude)))
        #The functions for the Tabor and the Tektro awg are not compatible
        #to be corrected
        channel_driver.vpp = float(self.format_and_eval_string(self.
                                                                 amplitude))