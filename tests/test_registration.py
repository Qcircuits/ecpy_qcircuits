# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015 by Ecpy_ext_demo Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""The version information for this release of Ecpy_ext_demo.

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

import enaml
from enaml.workbench.api import Workbench
from ecpy.testing.util import signal_error_raise

with enaml.imports():
    from enaml.workbench.core.core_manifest import CoreManifest
    from ecpy.app.packages.manifest import PackagesManifest
    from ecpy.app.errors.manifest import ErrorsManifest

pytest_plugins = str('ecpy.testing.fixtures'),


def test_registration(windows):
    """Test that the manifest is properly regsistered.

    """
    w = Workbench()
    w.register(CoreManifest())
    w.register(ErrorsManifest())
    w.register(PackagesManifest())

    with signal_error_raise():
        w.get_plugin('ecpy.app.packages').collect_and_register()

    # Edit the name of the package
    assert w.get_plugin('ecpy_ext_demo')
