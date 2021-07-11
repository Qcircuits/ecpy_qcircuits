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
from atom.api import (Str,Enum,Typed)
from exopy.utils.atom_util import ordered_dict_from_pref, ordered_dict_to_pref
from collections import OrderedDict
from exopy_hqc_legacy.instruments.drivers.visa_tools import InstrIOError


class AWGSetDCOffsetTask(InstrumentTask):
    """ Set the DC offset voltage of a given AWG channel

    """    
    
    channel = Str('1').tag(pref=True)
    voltage = Str('0').tag(pref=True)

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
    
    channel = Str('1').tag(pref=True)
    amplitude = Str('0').tag(pref=True)

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
        
class AWGSetMarkerTask(InstrumentTask):
    """ Set Markers of a given AWG channel

    """    
    
    Marker_Dict = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                    ordered_dict_from_pref))

    def perform(self):
        """
        """
        if not self.driver:
            self.start_driver()
        for l, v in self.Marker_Dict.items():
            channel_driver = self.driver.get_channel(int(l[2]))
            if l[-1] == '1':
                channel_driver._AWG.write("SOURce{}:MARK1:VOLTage:HIGH {}"
                            .format(channel_driver._channel, float(self.format_and_eval_string(v[0]))))
                channel_driver._AWG.write("SOURce{}:MARK1:VOLTage:LOW {}"
                            .format(channel_driver._channel, float(self.format_and_eval_string(v[1]))))
            else:
                channel_driver._AWG.write("SOURce{}:MARK2:VOLTage:HIGH {}"
                            .format(channel_driver._channel, float(self.format_and_eval_string(v[0]))))
                channel_driver._AWG.write("SOURce{}:MARK2:VOLTage:LOW {}"
                            .format(channel_driver._channel, float(self.format_and_eval_string(v[1]))))   
            
        for l, v in self.Marker_Dict.items():
            channel_driver = self.driver.get_channel(int(l[2]))
            if l[-1] == '1':
                result = channel_driver._AWG.ask_for_values("SOURce{}:MARK1:VOLTage:HIGH?"
                                                      .format(channel_driver._channel))[0]
                if abs(result - float(self.format_and_eval_string(v[0]))) > 10**-12:
                    raise InstrIOError(cleandoc('''AWG channel {} did not set
                                                    correctly the marker1 high
                                                    voltage'''.format(channel_driver._channel)))
                result = channel_driver._AWG.ask_for_values("SOURce{}:MARK1:VOLTage:LOW?"
                                                      .format(channel_driver._channel))[0]
                if abs(result - float(self.format_and_eval_string(v[1]))) > 10**-12:
                    raise InstrIOError(cleandoc('''AWG channel {} did not set
                                                    correctly the marker1 low
                                                    voltage'''.format(channel_driver._channel)))
            else:
                result = channel_driver._AWG.ask_for_values("SOURce{}:MARK2:VOLTage:HIGH?"
                                                      .format(channel_driver._channel))[0]
                if abs(result - float(self.format_and_eval_string(v[0]))) > 10**-12:
                    raise InstrIOError(cleandoc('''AWG channel {} did not set
                                                    correctly the marker2 high
                                                    voltage'''.format(channel_driver._channel)))
                result = channel_driver._AWG.ask_for_values("SOURce{}:MARK2:VOLTage:LOW?"
                                                      .format(channel_driver._channel))[0]
                if abs(result - float(self.format_and_eval_string(v[1]))) > 10**-12:
                    raise InstrIOError(cleandoc('''AWG channel {} did not set
                                                    correctly the marker2 low
                                                    voltage'''.format(channel_driver._channel)))
                
                
                
