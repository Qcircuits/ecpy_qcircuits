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


import logging

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import scipy.odr as odr
import numbers
import scipy.optimize as opt

from atom.api import (Str, set_default, Bool, Float, Str)
from exopy.tasks.api import SimpleTask, validators, TaskInterface, InterfaceableTaskMixin
from exopy.tasks.api import validators

from inspect import cleandoc


ARR_VAL = validators.Feval(types=np.ndarray)
VAL_REAL = validators.Feval(types=numbers.Real)

DEFAULT_FORMULA = \
'''def feedback_function(x):
    return np.sqrt(abs(x**2-157.46112542**2)/1796.56077597**2)'''

class FluxoniumFluxMapArrayTask(SimpleTask):
    """
    Returns the voltage and frequency arrays necessary for a sweep of the Fluxonium spectrum
    """
    
    #Fluxonium parameters
    Ec = Str().tag(pref=True)
    Ej = Str().tag(pref=True)
    El = Str().tag(pref=True)
    
    # number of fluxes to calculate
    N_calculated_fluxes = Str().tag(pref=True)
    
    #number of volts per flux quantum
    fluxquantum_volts = Str().tag(pref=True)
    
    #number of volts at half flux
    fluxhalf_volts = Str().tag(pref=True)
    
    #measure from volt_min to volt_max
    volt_min = Str().tag(pref=True)
    volt_max = Str().tag(pref=True)
    
    #measure from freq_min to freq_max
    freq_min = Str().tag(pref=True)
    freq_max = Str().tag(pref=True)
    
    #measure in frequency band around theoretical frequency
    freq_band = Str().tag(pref=True)
    
    #frequency and voltage steps
    freq_granularity = Str().tag(pref=True)
    volt_granularity = Str().tag(pref=True)
    
    #qubit transition to follow
    transition = Str().tag(pref=True)
    
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

class FitRamseyTask(SimpleTask):
    """ 
    Fit a post-selected Ramsey sequence and return detuning from drive frequency
    The sequence is RO, pi/2, wait, pi/2, RO
    The analysis makes lots of assumptions on the structure of the RO
    """
    
    target_data = Str().tag(pref=True, feval=ARR_VAL)
    
    wait = set_default({'activated': True}) # Wait on all pools by default.
    
    # keys of the data which we want to analyze
    time_key = Str().tag(pref=True)
    I_key = Str().tag(pref=True)
    Q_key = Str().tag(pref=True)
    show_plots = Bool(False).tag(pref=True)
    
    database_entries = set_default({'detuning': 0.0})
    
    def IQ_rot(self,data,theta=None):
        if theta != None:
            
            data_c = data * np.exp(1j *theta)
            return data_c
        
        else:
            
            dataf = data.flatten()
            I=np.real(dataf)
            Q=np.imag(dataf)
            Cov=np.cov(I,Q)
            A=np.linalg.eig(Cov)
            eigvecs=A[1]
            
            if A[0][1]>A[0][0]:
                eigvec1=eigvecs[:,0]
            else:
                eigvec1=eigvecs[:,1]
                
            theta=np.arctan(eigvec1[0]/eigvec1[1])
            data_c = data * np.exp(1j *theta)
    
            return data_c , theta
        
    def fit_and_show(self,x,y,func,p0,show_plot=True,show_guess=True,show_fit=True,show_points=True,ret_error=False,**kwargs):
    # func must be of the form func(p,x) where x is a data array and p is an array of parameters
        model = odr.Model(func)
        data = odr.Data(x,y)
        fit_object = odr.ODR(data,model,p0)
        output = fit_object.run()
        popt = output.beta
            
        if show_plot:
            xx = np.linspace(x[0],x[-1],1001)
            fit = func(popt,xx)
            initial = func(p0,xx)
            plt.figure()
            if show_guess:
                plt.plot(xx,initial,'g',label='initial')
            if show_points:
                plt.plot(x,y,label='signal',**kwargs)
            if show_fit:
                plt.plot(xx,fit,'r--',label='fit')
            plt.legend()
            plt.show()
            
        if ret_error:
            return p0, popt, output.sd_beta
        else:
            return p0, popt
        
    def Ramsey(self,p,x):
        f, A, b, phi = p
        return A*np.sin(2*np.pi*(f*x+phi))+b

    def perform(self):
        """ Fit a post-selected Ramsey sequence and return detuning from drive frequency
        """
        
        
        data = self.format_and_eval_string(self.target_data)
        t_ramsey = np.unique(np.array(data[self.time_key]))
        RO = np.swapaxes(1000*(np.array(data[self.I_key])+1j*np.array(data[self.Q_key])).reshape(len(t_ramsey),-1,2),-1,-2)
        

        '''
        ### code for RO with 3 blobs
        
        coordinates = np.stack((np.real(RO),np.imag(RO)),axis=-1)
        
        # find cluster centres and centre of circle which goes through those points
        # https://de.wikipedia.org/wiki/Umkreis
        clusters_fit = GaussianMixture(n_components=3,covariance_type='spherical').fit(coordinates[0,0])
        centres = clusters_fit.means_
        centres_cmplx = centres[:,0]+1j*centres[:,1]
        ccf = -centres[:,0]+1j*centres[:,1] # just a trick to shorten code
        d = 2*np.dot(centres[:,0],[centres[1,1]-centres[2,1],centres[2,1]-centres[0,1],centres[0,1]-centres[1,1]])
        circle_centre_inv = (abs(ccf[0])**2*(ccf[1]-ccf[2])+abs(ccf[1])**2*(ccf[2]-ccf[0])+abs(ccf[2])**2*(ccf[0]-ccf[1]))/d
        circle_centre = np.imag(circle_centre_inv)+1j*np.real(circle_centre_inv) # see wikipedia
        
        # use this to find angle of ground state and determine radius of post-selected distribution
        angles = np.unwrap(np.angle(centres_cmplx-circle_centre))
        ground_index = np.argmax(angles) #assumption about the position of the g-g state
        variance = clusters_fit.covariances_[ground_index]
        z_state = centres_cmplx[ground_index]
        r_select = 2*np.sqrt(variance) # 95% of data
        
        # plot for check
        if self.show_plots:
            xmin = np.amin(np.real(RO))
            xmax = np.amax(np.real(RO))
            ymin = np.amin(np.imag(RO))
            ymax = np.amax(np.imag(RO))
            xbins, xstep = np.linspace(xmin,xmax,101,retstep=True)
            ybins, ystep = np.linspace(ymin,ymax,101,retstep=True)
            xcentres = xbins[1:]-xstep/2
            ycentres = ybins[1:]-ystep/2
            
            xx = np.linspace(0,1,101)*np.pi*2
            r_state = 3*np.sqrt(variance) # 99.7% of data
            circ_base = np.cos(xx)+1j*np.sin(xx)
            circ_select = r_select*circ_base+z_state
            circ_state = r_state*circ_base+z_state
            hist_ref = np.histogramdd(coordinates[-1,-1],bins=(xbins,ybins),density=True)
            
            plt.figure()
            plt.pcolormesh(xcentres,ycentres,np.log10(hist_ref[0]).T)
            plt.plot(np.real(circ_state),np.imag(circ_state),c='purple')
            plt.plot(np.real(circ_select),np.imag(circ_select),c='r')
            plt.axes().set_aspect('equal')
            plt.show()
        
        in_zone = np.asarray([abs(RO[j,0]-z_state)<r_select for j in range(len(t_ramsey))])
        in_RO = [[RO[j,k][in_zone[j]] for k in range(2)] for j in range(len(t_ramsey))]
        in_RO2_avrg = self.IQ_rot(np.asarray([np.average(in_RO[j][1]) for j in range(len(t_ramsey))]))[0]
        in_RO2_avrg_zeroed = np.real(in_RO2_avrg)-np.average(np.real(in_RO2_avrg))
        
        # we remove the first point in the fit to avoid weird Alazar acquisition properties
        data_y = in_RO2_avrg_zeroed[1:]
        data_x = t_ramsey[1:]
        #guess the amplitude by comparing min and max
        ampl_0 = np.sign(data_y[1]-data_y[0])*(np.amax(data_y)-np.amin(data_y))/2
        #guess the frequency using fft
        freq_0 = np.fft.fftfreq(len(data_y))[np.argmax(abs(np.fft.fft(data_y)))]/(data_x[1]-data_x[0])
        
        
        #phase should not be necessary but there is some incomprehension about the fluxonium drives
        # guess the phase, could be off by the pi/2 complement
        phi_0 = np.arcsin(data_y[0]/ampl_0)/(2*np.pi)-freq_0*data_x[0]
        #correct guess based on evolution of first two points
        if (data_y[1]-data_y[0])<0:
            p0 = [freq_0,ampl_0,0,np.pi-phi_0]
        else:
            p0 = [freq_0,ampl_0,0,phi_0]
        
        # to make fit robust to single outliers we fit once
        # then remove the point furthest away from the fit
        # and fit again using the first fit as a guess
        popt_init = self.fit_and_show(data_x,data_y,self.Ramsey,p0,
                                      show_plot=False,
                                      ret_error=True)[1]
        
        dists = abs((self.Ramsey(popt_init,data_x)-data_y)**2)
        furthest_index = np.argmax(dists)
        data_y_nooutlier = np.delete(data_y,furthest_index)
        data_x_nooutlier = np.delete(data_x,furthest_index)
        
        popt_final,error = self.fit_and_show(data_x_nooutlier,data_y_nooutlier,
                                      self.Ramsey,popt_init,
                                      show_plot=self.show_plots,
                                      ret_error=True)[1:]
        
        if (error[0] > 1e-4) or (error[1:]> 1e-1).any():
            log = logging.getLogger(__name__)
            msg = ('Ramsey fit has abnormally high fit error.')
            log.warning(msg)
            raise ValueError(msg)
            
        measured_detuning = popt_final[0]*1000 # convert to MHz
        print('Ramsey detuning is: {}'.format(measured_detuning))
        self.write_in_database('detuning', measured_detuning)
        '''
        
        ### code for RO with 4 blobs
        RO = self.IQ_rot(RO)[0]
        
        xmin = np.amin(np.real(RO))
        xmax = np.amax(np.real(RO))
        ymin = np.amin(np.imag(RO))
        ymax = np.amax(np.imag(RO))
        xbins, xstep = np.linspace(xmin,xmax,101,retstep=True)
        ybins, ystep = np.linspace(ymin,ymax,101,retstep=True)
        xcentres = xbins[1:]-xstep/2
        ycentres = ybins[1:]-ystep/2
        
        coordinates = np.stack((RO[0,0].real,RO[0,0].imag),axis=-1)
        hist_ref = np.histogramdd(coordinates,bins=(xbins,ybins),density=True)
        
        tot_point_num = np.sum(np.nonzero(hist_ref[0]))
        weight_per_point = len(xcentres)*len(ycentres)/tot_point_num
        threshholded_hist_indices = np.argwhere(hist_ref[0]*xstep*ystep*weight_per_point>5/tot_point_num)
        
        RO_indices_r = ((coordinates[:,0]-xmin)//xstep).astype(int)
        RO_indices_r[RO_indices_r==100] = 99
        RO_indices_i = ((coordinates[:,1]-ymin)//ystep).astype(int)
        RO_indices_i[RO_indices_i==100] = 99
        RO_indices = np.stack((RO_indices_r,RO_indices_i),axis=-1).reshape(-1,2)
        
        valid_real = np.logical_and(RO_indices_r>=np.amin(threshholded_hist_indices[:,0]),RO_indices_r<=np.amax(threshholded_hist_indices[:,0]))
        valid_imag = np.logical_and(RO_indices_i>=np.amin(threshholded_hist_indices[:,1]),RO_indices_i<=np.amax(threshholded_hist_indices[:,1]))
        valid_indexes = np.logical_and(valid_real,valid_imag)
        
        ind = np.lexsort((RO_indices_i.flatten(),RO_indices_r.flatten()))
        sorted_RO_indices = RO_indices[ind]
        sorted_valid_indices = valid_indexes.flatten()[ind]
        
        current_threshed_index = 0
        flag = False
        helper_arr = np.zeros_like(sorted_RO_indices[:,0])
        
        for j,index in enumerate(sorted_RO_indices):
            if sorted_valid_indices[j]:
                if not flag:
                    if tuple(index) == tuple(threshholded_hist_indices[current_threshed_index]):
                        flag = True
                        helper_arr[j] = 1
                    else:
                        helper_arr[j] = 0
                else:
                    if tuple(index) == tuple(threshholded_hist_indices[current_threshed_index]):
                        helper_arr[j] = 1
                    else:
                        current_threshed_index += 1
                        if current_threshed_index >= len(threshholded_hist_indices):
                            break
                        if tuple(index) == tuple(threshholded_hist_indices[current_threshed_index]):
                            helper_arr[j] = 1
                        else:
                            flag = False
                            helper_arr[j] = 0
                        
            else:
                helper_arr[j] = 0 
        
        helper_arr_orig_order = helper_arr[np.argsort(ind)]
        
        threshed_RO = RO[0,0].flatten()[helper_arr_orig_order.astype(bool)]
        threshed_coords = np.stack((np.real(threshed_RO),np.imag(threshed_RO)),axis=-1)
        threshed_hist = np.histogramdd(threshed_coords,bins=(xbins,ybins),density=True)[0]
        
        clusters_fit = GaussianMixture(n_components=4,covariance_type='tied').fit(threshed_coords)
        centres = clusters_fit.means_[:,0]+1j*clusters_fit.means_[:,1]
        
        def circ_res(p):
            x0, y0, r = p
            return np.sum(abs(abs(centres-(x0+1j*y0))-r)**2)
        
        opt_circ_p = opt.minimize(circ_res,[np.real(centres[0])/2,np.imag(centres[0])/2,abs(centres[0])]).x
        circ_centre = opt_circ_p[0]+1j*opt_circ_p[1]
        
        angles = np.unwrap(np.angle(centres-circ_centre))
        indexes = np.argsort(angles)
        ground_index = indexes[-1]
        
        variance = np.mean(np.diag(clusters_fit.covariances_))
        z_state = centres[ground_index]
        r_select = 3*np.sqrt(variance) # 99.7% of data
        
        xx = np.linspace(0,1,101)*np.pi*2
        circ_base = np.cos(xx)+1j*np.sin(xx)
        res_circ = circ_base*opt_circ_p[-1]+circ_centre
        circ_select = r_select*circ_base+z_state
        
        if self.show_plots:
            fig, ax = plt.subplots(1,2)
            ax[0].pcolormesh(xcentres,ycentres,np.log10(threshed_hist).T)
            ax[1].pcolormesh(xcentres,ycentres,np.log10(hist_ref[0]).T)
            ax[1].plot(np.real(circ_select),np.imag(circ_select),'r')
            for axis in ax:
                axis.scatter(clusters_fit.means_[:,0],clusters_fit.means_[:,1],c='orange')
                axis.plot(np.real(res_circ),np.imag(res_circ))
                axis.set_aspect('equal')
            plt.show()
            
        in_zone = np.asarray([abs(RO[j,0]-z_state)<r_select for j in range(len(t_ramsey))])
        in_RO = [[RO[j,k][in_zone[j]] for k in range(2)] for j in range(len(t_ramsey))]
        in_RO2_avrg = self.IQ_rot(np.asarray([np.average(in_RO[j][1]) for j in range(len(t_ramsey))]))[0]
        in_RO2_avrg_zeroed = np.real(in_RO2_avrg)-np.average(np.real(in_RO2_avrg))
        
        def Ramsey(p,x):
            f,T,A,phi,b,c = p
            return A*np.exp(-x/T)*np.cos(2*np.pi*f*x+phi)+b+c*x
    
        data_y = in_RO2_avrg_zeroed[1:]
        data_x = t_ramsey[1:]*1e9
        ampl_0 = np.sign(data_y[1]-data_y[0])*(np.amax(data_y)-np.amin(data_y))/2
        #freq_0 = np.fft.fftfreq(len(data_y))[np.argmax(abs(np.fft.fft(data_y)))]/(data_x[1]-data_x[0])
        freq_0 = 1/400
        
        #phase should not be necessary but there is some incomprehension about the fluxonium drives
        if (data_y[1]-data_y[0])<0:
            p0 = [freq_0,500,-ampl_0,np.pi/2,0,1e-4]
        else:
            p0 = [freq_0,500,ampl_0,np.pi/2,0,1e-4]
        
        # to make fit robust to single outliers we fit once
        # then remove the point furthest away from the fit
        # and fit again using the first fit as a guess
        popt_init = self.fit_and_show(data_x,data_y,Ramsey,p0,show_plot=False,ret_error=True)[1]
        dists = abs((Ramsey(popt_init,data_x)-data_y)**2)
        furthest_index = np.argmax(dists)
        data_y_nooutlier = np.delete(data_y,furthest_index)
        data_x_nooutlier = np.delete(data_x,furthest_index)
        
        popt_final,error = self.fit_and_show(data_x_nooutlier,data_y_nooutlier,
                                      Ramsey,popt_init,
                                      show_plot=self.show_plots,
                                      ret_error=True)[1:]
        
        #if (abs(error/popt_final)[:-3] > 0.5).any():
        #    log = logging.getLogger(__name__)
        #    msg = ('Ramsey fit has abnormally high fit error.')
        #    log.warning(msg)
        #    raise ValueError(msg)
            
        measured_detuning = popt_final[0]*1000 # convert to MHz
        print('Ramsey detuning is: {}'.format(measured_detuning))
        self.write_in_database('detuning', measured_detuning)
        
    def check(self, *args, **kwargs):
        """ Check the target array can be found and has the right column.

        """

        return True, {}

    def _post_setattr_mode(self, old, new):
        """ Update the database entries according to the mode.

        """
        
class FindFluxDetuningTask(SimpleTask):
    """
    Takes a frequency detuning as input and based on model (also with input)
    tries to calculate a good guess for the expected flux point
    """
       
    measured_detuning = Str().tag(pref=True,feval=VAL_REAL)
    
    coupling = Str().tag(pref=True,feval=VAL_REAL)
    
    freq_to_volt = Str().tag(pref=True,feval=VAL_REAL)
    
    drive_freq = Str().tag(pref=True,feval=VAL_REAL)
    
    max_delta = Str().tag(pref=True,feval=VAL_REAL)
    
    database_entries = set_default({'current_freq': 157.0,
                                    'voltage_shift': 0.01,
                                    'target_frequency':157.0})
    
    
    
    def perform(self, coupling=None, freq_to_volt=None, measured_detuning=None, drive_freq=None, max_delta=None):
        
        if coupling is None:
            coupling = self.format_and_eval_string(self.coupling)
        if freq_to_volt is None:
            freq_to_volt = self.format_and_eval_string(self.freq_to_volt)
        if measured_detuning is None:
            measured_detuning = self.format_and_eval_string(self.measured_detuning)
        if drive_freq is None:
            drive_freq = self.format_and_eval_string(self.drive_freq)
        if max_delta is None:
            max_delta = self.format_and_eval_string(self.max_delta)
            
        def flux_delta(x): 
            return np.sqrt(abs(x**2-coupling**2)/freq_to_volt**2)
        
        current_freq = drive_freq + measured_detuning # we always fix the drive frequency below the achievable frequency minimum
        
        voltage_shift = np.round(flux_delta(current_freq),4)

        if voltage_shift > max_delta:
            log = logging.getLogger(__name__)
            msg = ('The calculated voltage shift is too large!')
            log.warning(msg)
            raise ValueError(msg)
            
        print('Frequency according to Ramsey is: {}'.format(current_freq))
        print('The suggested voltage shift is: {}'.format(voltage_shift))
        
        self.write_in_database('current_freq', current_freq)
        self.write_in_database('voltage_shift', voltage_shift)
        self.write_in_database('target_frequency', coupling)
        
class FindFluxDetuning2Task(SimpleTask):
    """
    Takes a frequency detuning as input and based on model (also with input)
    tries to calculate a good guess for the expected flux point
    """
    
    feedback_function_code = Str(DEFAULT_FORMULA).tag(pref=True)
    
    voltage_precision = Str().tag(pref=True,feval=VAL_REAL)
    
    measured_detuning = Str().tag(pref=True,feval=VAL_REAL)
    
    drive_freq = Str().tag(pref=True,feval=VAL_REAL)
    
    target_freq = Str().tag(pref=True,feval=VAL_REAL)
    
    max_delta = Str().tag(pref=True,feval=VAL_REAL)
    
    database_entries = set_default({'current_freq': 157.0,
                                    'voltage_shift': 0.01,
                                    'target_frequency':157.0})
    
    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super(FindFluxDetuning2Task, self).check(*args,
                                                             **kwargs)
        
        try:
            exec(self.feedback_function_code)
        except:
            test = False
            print('''The code-string used to define the feedback function
                     contains an error. Please check and test it again.''')
            return test, traceback
        
        try:
            locals()['feedback_function']
        except NameError as e:
            test = False
            print(e)
            print('''The name of the function used for the feedback
                     function is not valid. It should start as "def feedback_function(x):" 
                     where x is a frequency and it returns a voltage. ''')
            return test, traceback
            
        return test, traceback
        
    
    def perform(self, voltage_precision=None, measured_detuning=None, drive_freq=None, target_freq=None, max_delta=None):
        
        if voltage_precision is None:
            voltage_precision = self.format_and_eval_string(self.voltage_precision)
        if measured_detuning is None:
            measured_detuning = self.format_and_eval_string(self.measured_detuning)
        if drive_freq is None:
            drive_freq = self.format_and_eval_string(self.drive_freq)
        if target_freq is None:
            target_freq = self.format_and_eval_string(self.target_freq)
        if max_delta is None:
            max_delta = self.format_and_eval_string(self.max_delta)
            
        exec(self.feedback_function_code) ## creates a function named feedback_function
        
        current_freq = drive_freq + measured_detuning # we assume the drive frequency is below the qubit frequency
        
        current_voltage = np.round(locals()['feedback_function'](current_freq),voltage_precision)
        target_voltage = np.round(locals()['feedback_function'](target_freq),voltage_precision)
        
        voltage_shift = current_voltage-target_voltage
        
        print('Frequency according to Ramsey is: {}'.format(current_freq))
        print('The suggested voltage shift is: {}'.format(voltage_shift))

        if abs(voltage_shift) > max_delta:
            log = logging.getLogger(__name__)
            msg = ('The calculated voltage shift is too large!')
            log.warning(msg)
            raise ValueError(msg)
        
        self.write_in_database('current_freq', current_freq)
        self.write_in_database('voltage_shift', voltage_shift)
        self.write_in_database('target_frequency', target_freq)