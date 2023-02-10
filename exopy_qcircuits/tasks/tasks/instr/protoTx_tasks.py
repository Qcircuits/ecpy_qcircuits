# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2022 by Qcircuits Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Task to configure the protoTx

"""

from atom.api import Bool, Str, Enum, Float

from exopy.tasks.api import (InstrumentTask, validators)

class protoTxConfigurationTask(InstrumentTask):
    """Configure the protoTx

    """
    I_offset = Str("0").tag(pref=True)
    Q_offset = Str("0").tag(pref=True)
    operating_mode = Enum("IQ", "Lower", "Upper", "Synthesizer").tag(pref=True)
    reference_source = Enum("Internal", "External").tag(pref=True)
    lo_nulling = Bool(True).tag(pref=True)
    lo_source = Enum("Internal", "External").tag(pref=True)


    def check(self, *args, **kwargs):
        """Checks if the instrument is connected

        """
        test, traceback = super(protoTxConfigurationTask, self).check(*args, **kwargs)
        return test, traceback

    def perform(self):
       
        I_offset = int(self.I_offset)
        Q_offset = int(self.Q_offset)
        if self.operating_mode == "IQ":
            operating_mode = self.driver.OperatingMode.IQ
        elif self.operating_mode == "Upper":
            operating_mode = self.driver.OperatingMode.UPPER
        elif self.operating_mode == "Lower":
            operating_mode = self.driver.OperatingMode.LOWER
        else:
            operating_mode = self.driver.OperatingMode.SYNTHETIZER
        # self.driver.operating_mode = operating mode

        if self.reference_source == "Internal":
            reference_source = self.driver.Source.INTERNAL
        else:
            reference_source = self.driver.Source.EXTERNAL

        if self.lo_source == "Internal":
            lo_source = self.driver.Source.INTERNAL
        else:
            lo_source = self.driver.Source.EXTERNAL

        if not self.lo_nulling:
            # self.driver.lo_nulling = self.driver.LOnulling.FACTORY
            lo_nulling = self.driver.LOnulling.FACTORY
        else:
            lo_nulling = self.driver.LOnulling.MANUAL

        # self.driver.operating_mode = operating_mode
        # self.driver.reference_source = reference_source
        # self.driver.lo_source = lo_source
        # self.driver.Q_offset = Q_offset
        # self.driver.I_offset = I_offset
        self.driver.set_all_config(I_offset, Q_offset, operating_mode, reference_source, lo_nulling, lo_source)

class protoTxSetRFAttenuationTask(InstrumentTask):
    """Set RF attenuation

    """
    rf_attenuation = Str("0").tag(pref=True)



    def check(self, *args, **kwargs):
        """Checks if the instrument is connected

        """
        test, traceback = super(protoTxSetRFAttenuationTask, self).check(*args, **kwargs)
        return test, traceback

    def perform(self):

        self.driver.rf_attenuation = float(self.rf_attenuation)
