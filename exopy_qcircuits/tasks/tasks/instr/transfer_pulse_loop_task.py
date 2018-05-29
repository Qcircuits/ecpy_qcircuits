# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2016 by EcpyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Task to transfer a sequence on an AWG.

"""
import os
from traceback import format_exc
from pprint import pformat
from collections import OrderedDict
import numpy as np
import time
from atom.api import Value, Unicode, Float, Typed, Bool
from exopy.tasks.api import (InstrumentTask)
from exopy.utils.atom_util import ordered_dict_from_pref, ordered_dict_to_pref


class TransferPulseLoopTask(InstrumentTask):
    """Build and transfer a pulse sequence to an instrument.

    """
    #: Sequence path for the case of sequence simply referenced.
    sequence_path = Unicode().tag(pref=True)

    #: Time stamp of the last modification of the sequence file.
    sequence_timestamp = Float().tag(pref=True)

    #: Sequence of pulse to compile and transfer to the instrument.
    sequence = Value()

    #: Global variable to use for the sequence.
    sequence_vars = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                     ordered_dict_from_pref))

    #: Loop variables: channels on which the loop will be done, loop parameters
    #: names, start value, stop value and number of points per loop

    loop_name = Unicode('pulse_rabi_length').tag(pref=True)

    loop_start = Unicode('0').tag(pref=True)

    loop_stop = Unicode('1').tag(pref=True)

    loop_points = Unicode('2').tag(pref=True)

    #: internal or external trigger
    internal_trigger = Bool(False).tag(pref=True)

    #: Internal trigger period in mus
    trigger_period = Unicode('20').tag(pref=True)

    def check(self, *args, **kwargs):
        """Check that the sequence can be compiled.

        """
        test, traceback = super(TransferPulseLoopTask,
                                self).check(*args, **kwargs)
        err_path = self.path + '/' + self.name + '-'

        msg = 'Failed to evaluate {} ({}): {}'
        seq = self.sequence
        for k, v in self.sequence_vars.items():
            try:
                seq.external_vars[k] = self.format_and_eval_string(v)
            except Exception:
                test = False
                traceback[err_path+k] = msg.format(k, v, format_exc())

        if not test:
            return test, traceback

        context = seq.context
        res, infos, errors = context.compile_and_transfer_sequence(seq)

        if not res:
            traceback[err_path+'compil'] = errors
            return False, traceback

        for k, v in infos.items():
            self.write_in_database(k, v)

        if self.sequence_path:
            if not (self.sequence_timestamp ==
                    os.path.getmtime(self.sequence_path)):
                msg = 'The sequence is outdated, consider refreshing it.'
                traceback[err_path+'outdated'] = msg

        return test, traceback

    def perform(self):
        """Compile the sequence.

        """
        self.driver.run_mode = 'SEQUENCE'
        self.driver.delete_all_waveforms
        self.driver.clear_all_sequences
        time.sleep(10)
        seq = self.sequence
        context = seq.context
        context.run_after_transfer = False
        context.select_after_transfer = False
        loop_start = float(self.format_and_eval_string(self.loop_start))
        loop_stop = float(self.format_and_eval_string(self.loop_stop))
        loop_points = int(self.format_and_eval_string(self.loop_points))

        self.driver.internal_trigger = self.internal_trigger
        if self.internal_trigger:
            self.driver.internal_trigger_period = int(float(self.trigger_period) * 1000)

        loop_values = np.linspace(loop_start, loop_stop, loop_points)

        current_pos = 0
        for nn in range(loop_points):
            self.sequence_vars[self.loop_name] = str(loop_values[nn])
            for k, v in self.sequence_vars.items():
                seq.external_vars[k] = self.format_and_eval_string(v)
            context.sequence_name = '{}'.format(nn+1)
            res, byteseq, repeat, infos, errors = context.compile_loop(seq)


            already_added = {}
            for ch_id in self.driver.defined_channels:
                if ch_id in byteseq:
                    for pos,waveform in enumerate(byteseq[ch_id]):
                        addr = id(waveform)
                        if addr not in already_added:
                            seq_name_transfered = context.sequence_name  + '_Ch{}'.format(ch_id) +\
                                                '_' + str(pos)
                            self.driver.to_send(seq_name_transfered, waveform)
                            already_added[addr] = seq_name_transfered
                        else:
                            seq_name_transfered =  already_added[addr]
                        self.driver.get_channel(ch_id).set_sequence_pos(seq_name_transfered, current_pos + pos + 1)
                        self.driver.set_repeat(current_pos + pos + 1, repeat[pos])
                        self.driver.set_goto_pos(current_pos + pos + 1, current_pos + pos + 2)

            current_pos += len(byteseq[self.driver.defined_channels[0]])

        self.driver.set_goto_pos(current_pos, 1)

        for ch_id in self.driver.defined_channels:
           if ch_id in byteseq:
               ch = self.driver.get_channel(ch_id)
               ch.output_state = 'ON'


        if not res:
            raise Exception('Failed to compile sequence')

        for k, v in infos.items():
            self.write_in_database(k, v)

    def register_preferences(self):
        """Register the task preferences into the preferences system.

        """
        super(TransferPulseLoopTask, self).register_preferences()

        if self.sequence:
            self.preferences['sequence'] =\
                self.sequence.preferences_from_members()

    update_preferences_from_members = register_preferences

    def traverse(self, depth=-1):
        """Reimplemented to also yield the sequence

        """
        infos = super(TransferPulseLoopTask, self).traverse(depth)

        for i in infos:
            yield i

        for item in self.sequence.traverse():
            yield item

    @classmethod
    def build_from_config(cls, config, dependencies):
        """Rebuild the task and the sequence from a config file.

        """
        builder = cls.mro()[1].build_from_config.__func__
        task = builder(cls, config, dependencies)

        if 'sequence' in config:
            pulse_dep = dependencies['ecpy.pulses.item']
            builder = pulse_dep['ecpy_pulses.RootSequence']
            conf = config['sequence']
            seq = builder.build_from_config(conf, dependencies)
            task.sequence = seq

        return task

    def _post_setattr_sequence(self, old, new):
        """Set up n observer on the sequence context to properly update the
        database entries.

        """
        entries = self.database_entries.copy()
        if old:
            old.unobserve('context', self._update_database_entries)
            if old.context:
                for k in old.context.list_sequence_infos():
                    del entries[k]
        if new:
            new.observe('context', self._update_database_entries)
            if new.context:
                entries.update(new.context.list_sequence_infos())

        if entries != self.database_entries:
            self.database_entries = entries

    def _update_database_entries(self, change):
        """Reflect in the database the sequence infos of the context.

        """
        entries = self.database_entries.copy()
        if change.get('oldvalue'):
            for k in change['oldvalue'].list_sequence_infos():
                del entries[k]
        if change['value']:
            context = change['value']
            entries.update(context.list_sequence_infos())

        self.database_entries = entries
