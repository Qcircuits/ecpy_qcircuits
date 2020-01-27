# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2018 by ExopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Tasks to operate on numpy.arrays.

"""
import numpy as np
import qutip as qt
from atom.api import (Enum, Unicode, set_default)
from exopy.tasks.api import SimpleTask, validators


ARR_VAL = validators.Feval(types=np.ndarray)

class FluxoniumFluxMapArrayTask(SimpleTask):
    """
    Returns the voltage and frequency arrays necessary for a sweep of the Fluxonium spectrum
    """
    
    #Fluxonium parameters
    Ec = Unicode().tag(pref=True)
    Ej = Unicode().tag(pref=True)
    El = Unicode().tag(pref=True)
    
    # number of fluxes to calculate
    N_calculated_fluxes = Unicode().tag(pref=True)
    
    #number of volts per flux quantum
    fluxquantum_volts = Unicode().tag(pref=True)
    
    #number of volts at half flux
    fluxhalf_volts = Unicode().tag(pref=True)
    
    #measure from volt_min to volt_max
    volt_min = Unicode().tag(pref=True)
    volt_max = Unicode().tag(pref=True)
    
    #measure from freq_min to freq_max
    freq_min = Unicode().tag(pref=True)
    freq_max = Unicode().tag(pref=True)
    
    #measure in frequency band around theoretical frequency
    freq_band = Unicode().tag(pref=True)
    
    #frequency and voltage steps
    freq_granularity = Unicode().tag(pref=True)
    volt_granularity = Unicode().tag(pref=True)
    
    #qubit transition to follow
    transition = Unicode().tag(pref=True)
    
    database_entries = set_default({'volt_tomeasure': np.empty(2), 'freqs_tomeasure': np.empty((2,2))})
    
    def perform(self):
        # volts converted to fluxes (0 - 2pi) to calculate using qutip
        volt_tocalc, volt_calcstep = np.linspace(self.format_and_eval_string(self.volt_min),
                                                 self.format_and_eval_string(self.volt_max),
                                                 self.format_and_eval_string(self.N_calculated_fluxes),
                                                 retstep=True)
        
        
        flux_tocalc = (volt_tocalc-self.format_and_eval_string(self.fluxhalf_volts))/self.format_and_eval_string(self.fluxquantum_volts)*np.pi*2+np.pi
        
        # dimension of qubit hilbert space
        dim_q = 20
        #define qubit hamiltonian
        a = qt.destroy(dim_q)
        num = 1j*(a.dag()-a)/(np.sqrt(2*np.sqrt(8*self.format_and_eval_string(self.Ec)/self.format_and_eval_string(self.El))))
        phi = (a+a.dag())*(np.sqrt(np.sqrt(8*self.format_and_eval_string(self.Ec)/self.format_and_eval_string(self.El))/2))
        def Ham(phiext):
            cosphi = (phi+phiext*qt.qeye(dim_q)).cosm()
            return 4*self.format_and_eval_string(self.Ec)*num**2+0.5*self.format_and_eval_string(self.El)*phi**2-self.format_and_eval_string(self.Ej)*cosphi
        
        #calculate eigenenergies at each flux
        eigenenergies = np.array([Ham(phiext).eigenenergies() for phiext in flux_tocalc])
        #calculate the transitions between each pair of levels
        transitions = eigenenergies[:,np.newaxis, :] - eigenenergies[:,:, np.newaxis]
        
        #choose transition we want to follow
        lvl0, lvl1 = map(int,self.transition.split('-'))
        freq_max = transitions[:,lvl0,lvl1]+self.format_and_eval_string(self.freq_band)/2
        freq_min = transitions[:,lvl0,lvl1]-self.format_and_eval_string(self.freq_band)/2
        
        # define voltages we want to measure at
        volt_tomeasure = np.linspace(self.format_and_eval_string(self.volt_min),
                                     self.format_and_eval_string(self.volt_max),
                                     (self.format_and_eval_string(self.volt_max)-self.format_and_eval_string(self.volt_min))/self.format_and_eval_string(self.volt_granularity)+1)
        
        #find closest calculated flux point
        calc_indexes = np.array(np.round((volt_tomeasure-self.format_and_eval_string(self.volt_min))/volt_calcstep),dtype=int)
        
        # define frequencies to measure at based on calculation
        freqs_tomeasure = np.empty((len(volt_tomeasure),
                                    int(self.format_and_eval_string(self.freq_band)/self.format_and_eval_string(self.freq_granularity)+1)))
        for i,index in enumerate(calc_indexes):
            freqs_tomeasure[i] = np.linspace(freq_min[index],freq_max[index],
                                             self.format_and_eval_string(self.freq_band)/self.format_and_eval_string(self.freq_granularity)+1)
        
        #ensure frequency bounds are respected
        if self.freq_max != '':
            freqs_tomeasure[freqs_tomeasure>self.format_and_eval_string(self.freq_max)] = self.format_and_eval_string(self.freq_max)
        if self.freq_min != '':
            freqs_tomeasure[freqs_tomeasure<self.format_and_eval_string(self.freq_min)] = self.format_and_eval_string(self.freq_min)
        
        #write results to database
        self.write_in_database('volt_tomeasure', np.around(volt_tomeasure,3))
        self.write_in_database('freqs_tomeasure', np.around(freqs_tomeasure,int(np.ceil(-np.log10(self.format_and_eval_string(self.freq_granularity))))))
        
        #return results
        return volt_tomeasure, freqs_tomeasure