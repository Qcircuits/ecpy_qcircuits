# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2016 by exopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""The manifest contributing the extensions to the main application.

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

from exopy_hqc_legacy.instruments.starters.legacy_starter import LegacyStarter


class ZIQCircuitStarter(LegacyStarter):
    """Starter for ZI instruments.

    """
    def format_connection_infos(self, infos):
        """Rename serial_number to device_number.

        """
        i = infos.copy()
        i['device_number'] = infos['serial_number']
        del i['serial_number']
        return i
    
    def check_infos(self, driver_cls, connection, settings):
        """Attempt to open the connection to the instrument.

        """
        c = self.format_connection_infos(connection)
        c.update(settings)
        driver = None
        '''
        try:
            driver = driver_cls(c)
            res = driver.connected
        except Exception:
            return False, format_exc()
        finally:
            if driver is not None:
                driver.close_connection()
       < '''
        return True, ('Instrument does not appear to be connected but no '
                     'exception was raised.')