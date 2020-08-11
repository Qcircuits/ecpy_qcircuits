# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2016 by ExopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Task perform measurements with a ENA.

"""
import time
import re
import numbers
from inspect import cleandoc

import numpy as np
from atom.api import (Str, Int, Bool, Enum, set_default,
                      Value, List)

from exopy.tasks.api import InstrumentTask, TaskInterface, validators



def check_channels_presence(task, channels, *args, **kwargs):
    """ Check that all the channels are correctly defined on the PNA.

    """
    if kwargs.get('test_instr'):
        traceback = {}
        err_path = task.get_error_path()
        with task.test_driver() as instr:
            if instr is None:
                return False, traceback
            channels_present = True
            for channel in channels:
                if channel not in instr.defined_channels:
                    key = err_path + '_' + str(channel)
                    msg = ("Channel {} is not defined in the PNA {}."
                           " Please define it yourself and try again.")
                    traceback[key] = msg.format(channel,
                                                task.selected_instrument[0])

                    channels_present = False

            return channels_present, traceback

    else:
        return True, {}


class PNASetRFFrequencyInterface(TaskInterface):
    """Set the central frequecny to be used for the specified channel.

    """
    # Id of the channel whose central frequency should be set.
    channel = Int(1).tag(pref=True)

    # Driver for the channel.
    channel_driver = Value()

    has_view = True

    def perform(self, frequency=None):
        """
        """
        task = self.task
        if not task.driver:
            task.start_driver()
            self.channel_driver = task.driver.get_channel(self.channel)

        task.driver.owner = task.name
        self.channel_driver.owner = task.name

        if frequency is None:
            frequency = task.format_and_eval_string(task.frequency)
            frequency = task.convert(frequency, 'Hz')

        self.channel_driver.frequency = frequency
        task.write_in_database('frequency', frequency)

    def check(self, *args, **kwargs):
        """

        """
        task = self.task
        return check_channels_presence(task, [self.channel], *args, **kwargs)


class PNASetRFPowerInterface(TaskInterface):
    """Set the central power to be used for the specified channel.

    """
    # Id of the channel whose central frequency should be set.
    channel = Int(1).tag(pref=True)

    # Driver for the channel.
    channel_driver = Value()

    # Port whose output power should be set.
    port = Int(1).tag(pref=True)

    has_view = True

    def perform(self, power=None):
        """
        """
        task = self.task
        if not task.driver:
            task.start_driver()
            self.channel_driver = task.driver.get_channel(self.channel)

        task.driver.owner = task.name
        self.channel_driver.owner = task.name

        if power is None:
            power = task.format_and_eval_string(task.power)

        self.channel_driver.port = self.port
        self.channel_driver.power = power
        task.write_in_database('power', power)

    def check(self, *args, **kwargs):
        """

        """
        task = self.task
        return check_channels_presence(task, [self.channel], *args, **kwargs)


INTERFACES = {'SetRFFrequencyTask': [PNASetRFFrequencyInterface],
              'SetRFPowerTask': [PNASetRFPowerInterface]}


class SingleChannelPNATask(InstrumentTask):
    """ Helper class managing the notion of channel in the PNA.

    """
    # Id of the channel to use.
    channel = Int(1).tag(pref=True)

    channel_driver = Value()

    def check(self, *args, **kwargs):
        """ Add checking for channels to the base tests.

        """
        test, traceback = super(SingleChannelPNATask, self).check(*args,
                                                                  **kwargs)
        c_test, c_trace = check_channels_presence(self, [self.channel],
                                                  *args, **kwargs)

        traceback.update(c_trace)
        return test and c_test, traceback

FEVAL = validators.SkipEmpty(types=numbers.Real)


class ENASweepTask(SingleChannelPNATask):
    """Measure the specified parameters while sweeping either the frequency or
    the power. Measure are saved in an array with named fields : Frequency or
    Power and then 'Measure'_'Format' (S21_MLIN, S33 if Raw)

    Wait for any parallel operation before execution.

    """
    channel = Int(1).tag(pref=True)

    start = Str().tag(pref=True, feval=FEVAL)

    stop = Str().tag(pref=True, feval=FEVAL)

    points = Str().tag(pref=True, feval=FEVAL)
    
    sweep_type = Enum('','Frequency', 'Power').tag(pref=True)

    measures = List().tag(pref=True)

    if_bandwidth = Int(0).tag(pref=True)

    window = Int(1).tag(pref=True)

    wait = set_default({'activated': True, 'wait': ['instr']})

    database_entries = set_default({'sweep_data': np.array([0])})

    def perform(self):
        """
        """
        if not self.driver:
            self.start_driver()
            self.channel_driver = self.driver.get_channel(self.channel)
            self.channel_driver = self.driver.get_channel(self.channel)

        if self.driver.owner != self.name:
            self.driver.owner = self.name
            self.driver.set_all_chanel_to_hold()
            self.driver.trigger_scope = 'CURRent'
            self.driver.trigger_source = 'MANual'

        meas_names = ['Ch{}:'.format(self.channel) + ':'.join(measure)
                      for measure in self.measures]

        if self.channel_driver.owner != self.name:
            self.channel_driver.owner = self.name
            if self.if_bandwidth>0:
                self.channel_driver.if_bandwidth = self.if_bandwidth

            # Check whether or not we are doing the same measures as the ones
            # already defined (avoid losing display optimisation)
            measures = self.channel_driver.list_existing_measures()
            existing_meas = [meas['name'] for meas in measures]

            if not (all([meas in existing_meas for meas in meas_names])
                    and all([meas in meas_names for meas in existing_meas])):
                clear = True
                self.channel_driver.delete_all_meas()
                for i, meas_name in enumerate(meas_names):
                    self.channel_driver.prepare_measure(meas_name, self.window,
                                                        i+1, clear)
                    clear = False
        current_Xaxis = self.channel_driver.sweep_x_axis
        if self.start:
            start = self.format_and_eval_string(self.start)
        else:
            start = current_Xaxis[0]*1e9
        if self.stop:
            stop = self.format_and_eval_string(self.stop)
        else:
            stop = current_Xaxis[-1]*1e9
        if self.points:
            points = self.format_and_eval_string(self.points)
        else:
            points = len(current_Xaxis)
        if self.sweep_type:
            self.channel_driver.prepare_sweep(self.sweep_type.upper(), start,
                                              stop, points)
        else:
            if self.channel_driver.sweep_type.upper() == 'LIN':
                self.channel_driver.prepare_sweep('FREQUENCY',
                                                  start, stop, points)
            elif self.channel_driver.sweep_type.upper() == 'POW':
                 self.channel_driver.prepare_sweep('POWER',
                                                  start, stop, points)

        waiting_time = self.channel_driver.sweep_time
        self.driver.fire_trigger(self.channel)
        time.sleep(waiting_time)
        while not self.driver.check_operation_completion():
            time.sleep(0.1*waiting_time)

        data = [np.linspace(start, stop, points)]
        for i, meas_name in enumerate(meas_names):
            if self.measures[i][1]:
                data.append(
                    self.channel_driver.read_formatted_data(meas_name))
            else:
                data.append(self.channel_driver.read_raw_data(meas_name))

        names = [self.sweep_type] + ['_'.join(measure)
                                     for measure in self.measures]
        final_arr = np.rec.fromarrays(data, names=names)
        self.write_in_database('sweep_data', final_arr)

    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super(ENASweepTask, self).check(*args, **kwargs)

        pattern = re.compile('S[1-4][1-4]')
        for i, meas in enumerate(self.measures):
            match = pattern.match(meas[0])
            if not match:
                path = self.task_path + '/' + self.name
                path += '_Meas_{}'.format(i)
                traceback[path] = 'Unvalid parameter : {}'.format(meas[0])
                test = False
        if self.start:
            try:
                self.format_and_eval_string(self.start)
            except:
                test = False
                traceback[self.task_path + '/' + self.name + '-start'] = \
                    'Failed to eval the start formula {}'.format(self.start)
        if self.stop:
            try:
                self.format_and_eval_string(self.stop)
            except:
                test = False
                traceback[self.task_path + '/' + self.name + '-stop'] = \
                    'Failed to eval the stop formula {}'.format(self.stop)
        if self.points:
            try:
                self.format_and_eval_string(self.points)
            except:
                test = False
                traceback[self.task_path + '/' + self.name + '-step'] = \
                    'Failed to eval the points formula {}'.format(self.points)

        data = [np.array([0.0, 1.0])] + \
            [np.array([0.0, 1.0]) for meas in self.measures]
        names = [self.sweep_type] + ['_'.join(meas) for meas in self.measures]
        final_arr = np.rec.fromarrays(data, names=names)

        self.write_in_database('sweep_data', final_arr)
        return test, traceback

class ENAGetTracesTask(InstrumentTask):
    """ Get the traces that are displayed right now (no new acquisition).

    The list of traces to be measured must be entered in the following format
    ch1,tr1;ch2,tr2;ch3,tr3;...
    ex: 1,1;1,3 for ch1, tr1 and ch1, tr3

    """

    #tracelist = Str('1,1').tag(pref=True, feval=FEVAL)
    tracelist = Str('1,1').tag(pref=True)
    already_measured = Bool(False).tag(pref=True)

    database_entries = set_default({'sweep_data': {}})

    def perform(self):
        traces = self.tracelist.split(';')

        if not self.driver:
            self.start_driver()

        tr_data = {}

        if not self.already_measured:
            for i in range(1,30):
                if str(i)+',' in self.tracelist:
                    self.average_channel(i)

        for trace in traces:
            c_nb, t_nb = trace.split(',')
            tr_data[trace] = self.get_trace(int(c_nb), int(t_nb))

        self.write_in_database('sweep_data', tr_data)

    def average_channel(self, channelnb):
        """ Performs the averaging of a channel

        """
        channel_driver = self.driver.get_channel(channelnb)
        channel_driver.run_averaging()

    def get_trace(self, channelnb, tracenb):
        """ Get the trace that is displayed right now (no new acquisition)
        on channel and tracenb.

        """

        channel_driver = self.driver.get_channel(channelnb)

        try:
            channel_driver.tracenb = tracenb
        except:
            raise ValueError(cleandoc('''The trace {} does not exist on channel
                                      {}: '''.format(tracenb, channelnb)))

        measname = [channelnb,tracenb]
        data = channel_driver.sweep_x_axis
        complexdata = channel_driver.read_raw_data(measname)* \
                np.exp(2*np.pi*1j*data*channel_driver.electrical_delay)
        aux = [data, complexdata.real, complexdata.imag,
                np.absolute(complexdata),
                np.unwrap(np.angle(complexdata))]

        return np.rec.fromarrays(aux, names=['Freq (GHz)', str(measname)+' real',
                    str(measname)+' imag',  str(measname)+' abs',  str(measname)+' phase' ])

    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super(ENAGetTracesTask, self).check(*args, **kwargs)
        traces = self.tracelist.split(';')

        sweep_data = {}
        for trace in traces:
            data = [np.array([0.0, 1.0]), np.array([1.0, 2.0])]
            sweep_data[trace] = np.rec.fromarrays(data, names=['a', 'b'])

        self.write_in_database('sweep_data', sweep_data)
        return test, traceback


class GetMarkerPosition(InstrumentTask):
    """Determine the frequency of the marker

    """

    tracelist = Str('1,1').tag(pref=True, feval=FEVAL)

    has_view = True

    database_entries = set_default({'markerfreq': 1.0})

    def perform(self):
        """

        """

        traces = self.tracelist.split(';')
        if not self.driver:
            self.start_driver()


        for i in range(1,30):
            if str(i)+',' in self.tracelist:
                self.average_channel(i)

        c_nb, t_nb = traces[0].split(',')
        self.write_in_database('markerfreq',
                               self.get_marker(int(c_nb), int(t_nb)))

    def average_channel(self, channelnb):
        """ Performs the averaging of a channel

        """
        channel_driver = self.driver.get_channel(channelnb)
        channel_driver.run_averaging()

    def get_marker(self, channelnb, tracenb):
        """ Get the trace that is displayed right now (no new acquisition)
        on channel and tracenb.

        """

        channel_driver = self.driver.get_channel(channelnb)

        try:
            channel_driver.tracenb = tracenb
        except:
            raise ValueError(cleandoc('''The trace {} does not exist on channel
                                      {}: '''.format(tracenb, channelnb)))

# measname = channel_driver.selected_measure
        return channel_driver.marker_freq
