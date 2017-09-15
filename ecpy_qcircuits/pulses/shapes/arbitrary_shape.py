# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2017 by EcpyPulses Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Gaussian shapes

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

from numbers import Real

import numpy as np
from atom.api import Unicode, Callable, Str

from ecpy_pulses.pulses.utils.validators import Feval

from ..utils.entry_eval import exec_entry


from ecpy_pulses.pulses.shapes.base_shape import AbstractShape

DEFAULT_FORMULA = \
'''def c(self, time, unit):
    return 0.5*np.ones(len(time))'''

class ArbitraryShape(AbstractShape):
    """ Shape defined entirely by the user.

    """
    #: Formula used to compute the shape of the pulse. It is compiled as
    #: a function using exec which must be of the following signature:
    #: c(self, time, unit) and return the pulse amplitude as a numpy array.
    #: 'time' is a numpy array which represents the times at which to compute
    #: the pulse
    #: 'unit' is the unit in which the time is expressed.
    #: During compilation, all the sequence local variables can be accessed
    #: (using the {} notation).
    formula = Str(DEFAULT_FORMULA).tag(pref=True)
    

    def eval_entries(self, root_vars, sequence_locals, missing, errors):
        """ Evaluate the amplitude of the pulse.

        Parameters
        ----------
        root_vars : dict
            Global variables. As shapes and modulation cannot update them an
            empty dict is passed.

        sequence_locals : dict
            Known locals variables for the pulse sequence.

        missing : set
            Set of variables missing to evaluate some entries in the sequence.

        errors : dict
            Errors which occurred when trying to compile the pulse sequence.

        Returns
        -------
        result : bool
            Flag indicating whether or not the evaluation succeeded.

        """
        # Executing the formula :
        res, err = self.build_compute_function(sequence_locals, missing)
        
        return res
    
    
    def compute(self, time, unit):
        """ Computes the shape of the pulse at a given time.

        Parameters
        ----------
        time : ndarray
            Times at which to compute the modulation.

        unit : str
            Unit in which the time is expressed.

        Returns
        -------
        shape : ndarray
            Amplitude of the pulse.

        """
        shape = self._shape_factory(self, time, unit)
        assert np.max(shape) <= 1.0
        assert np.min(shape) >= -1.0
        return shape
    
    def build_compute_function(self, sequence_locals, missing):
        """Build the compute function from the formula.

        """
        try:
            loc = exec_entry(self.formula, sequence_locals, missing)
            if not loc:
                return False, {}
            self._shape_factory = loc['c']
            
        except Exception:
            return False, {}
        
        return True, {}
    
    # --- Private API ---------------------------------------------------------

    #: Runtime build shape computer.
    _shape_factory = Callable()