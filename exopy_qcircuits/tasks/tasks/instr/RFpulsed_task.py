# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2018 by Qcircuits Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Set the pulse mode ON/OFF on RF sources

"""

from atom.api import Bool

from exopy.tasks.api import (InstrumentTask, validators)

class SetRFPulsedTask(InstrumentTask):
    """Set the frequency of the signal delivered by a RF source.

    """
    # Whether to start the source in pulsed mode
    pulsed = Bool(False).tag(pref=True)

    def check(self, *args, **kwargs):
        """Checks if the instrument is connected

        """
        test, traceback = super(SetRFPulsedTask, self).check(*args,
                                                                **kwargs)
        return test, traceback

    def perform(self):
        """Default interface for simple sources.

        """
        if self.pulsed:
            self.driver.pm_state = 'ON'
        else:
            self.driver.pm_state = 'OFF'