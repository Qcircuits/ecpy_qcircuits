# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2016 by Qcircuits Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Compatibility package providing HQCMeas tasks and drivers in Exopy.

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)


def list_manifests():
    """List the manifest that should be regsitered when the main Exopy app is
    started.

    """
    import enaml

    with enaml.imports():
        from .manifest import QcircuitsManifest

    manifests = [QcircuitsManifest]

    return manifests
