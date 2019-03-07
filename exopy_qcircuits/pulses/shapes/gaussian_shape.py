# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2017 by ExopyPulses Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Gaussian shapes

"""
from numbers import Real

import numpy as np
from atom.api import Unicode

from exopy_pulses.pulses.utils.validators import Feval

from exopy_pulses.pulses.shapes.base_shape import AbstractShape


class GaussianShape(AbstractShape):
    """ Basic gaussian pulse with a variable amplitude.

    """

    amplitude = Unicode('1.0').tag(pref=True, feval=Feval(types=Real))

    width = Unicode('').tag(pref=True, feval=Feval(types=Real))

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
        res = super(GaussianShape, self).eval_entries(root_vars, sequence_locals,
                                                    missing, errors)

        if res:
            if not -1.0 <= self._cache['amplitude'] <= 1.0:
                msg = 'Shape amplitude must be between -1 and 1.'
                errors[self.format_error_id('amplitude')] = msg
                res = False
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
        return self._cache['amplitude']*np.exp(-np.square(time-(time[0]+time[-1])/2)/(2*self._cache['width']**2))


class GaussianEdgeShape(AbstractShape):
    """ Basic gaussian edge pulse with a variable amplitude.

    """

    amplitude = Unicode('1.0').tag(pref=True, feval=Feval(types=Real))

    edge_width = Unicode('').tag(pref=True, feval=Feval(types=Real))

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
        res = super(GaussianEdgeShape, self).eval_entries(root_vars, sequence_locals,
                                                    missing, errors)

        if res:
            if not -1.0 <= self._cache['amplitude'] <= 1.0:
                msg = 'Shape amplitude must be between -1 and 1.'
                errors[self.format_error_id('amplitude')] = msg
                res = False
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
        out = np.ones(len(time))
        c = self._cache['edge_width']/(2*np.sqrt(2*np.log(2)))
        gaussian = np.exp(-np.square(time-time[0]-2 * self._cache['edge_width'])/(2*c**2))
        out[time - time[0] <= self._cache['edge_width']*2] = gaussian[time-time[0] <= self._cache['edge_width']*2]
        out[time - time[0] >= len(time) - self._cache['edge_width']*2] = gaussian[(time-time[0]>self._cache['edge_width']*2)*(time-time[0]<=self._cache['edge_width']*4)]

        return self._cache['amplitude']*out
        
        
        
        

class SechShape(AbstractShape):
    """ Basic sech pulse with a variable amplitude.

    """

    amplitude = Unicode('1.0').tag(pref=True, feval=Feval(types=Real))

    width = Unicode('').tag(pref=True, feval=Feval(types=Real))

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
        res = super(SechShape, self).eval_entries(root_vars, sequence_locals,
                                                    missing, errors)

        if res:
            if not -1.0 <= self._cache['amplitude'] <= 1.0:
                msg = 'Shape amplitude must be between -1 and 1.'
                errors[self.format_error_id('amplitude')] = msg
                res = False
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
        # The sqrt(pi/2) factor is so that the fft of sec(x/sigma) is sec(x*sigma)
        return self._cache['amplitude']/np.cosh((time-(time[0]+time[-1])/2.0)*np.sqrt(np.pi/2)/(self._cache['width']))
    
class GaussienDerivativeShape(AbstractShape):
    """ Basic gaussian pulse with a variable amplitude.

    """

    amplitude = Unicode('1.0').tag(pref=True, feval=Feval(types=Real))

    width = Unicode('').tag(pref=True, feval=Feval(types=Real))
    

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
        res = super(GaussienDerivativeShape, self).eval_entries(root_vars, sequence_locals,
                                                    missing, errors)

        if res:
            if not -1.0 <= self._cache['amplitude'] <= 1.0:
                msg = 'Shape amplitude must be between -1 and 1.'
                errors[self.format_error_id('amplitude')] = msg
                res = False
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
        amp = self._cache['amplitude']
        sigma = self._cache['width']
        t0 = (time[0]+time[-1])/2
        pulse_shape = [-(t-t0)/(time[-1]-time[0])*4*amp*np.exp(-(t-t0)**2/2/sigma**2) for t in time]
        return np.asarray(pulse_shape)
