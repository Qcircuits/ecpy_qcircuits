# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2022 by exopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Interface for the multichannel E3631A voltage source

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)


from atom.api import Enum

from exopy.tasks.tasks.task_interface import TaskInterface


class E3631AInterface(TaskInterface):
    channel = Enum('P6V','P25V', 'N25V').tag(pref=True)

    def check(self, *args, **kwargs):
        return True, {}

    def perform(self, *args, **kwargs):
        self.task.driver.channel = self.channel

        return self.task.i_perform(*args, **kwargs)