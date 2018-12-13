# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2017 by exopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Interface allowing to use RF set power task with KeysightENA.

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

from atom.api import Int

from exopy.tasks.api import TaskInterface


class KeysightENAsetPowerInterface(TaskInterface):
    """Set the channel selected by user.

    """
    #: Id of the channel whose central frequency should be set.
    channel = Int(0).tag(pref=True)

    def perform(self, frequency=None):
        """Set the requested power to the selected channel

        """
        task = self.task
        channel = self.channel
        channel_driver = task.driver.get_channel(channel)
        channel_driver.power = task.power
