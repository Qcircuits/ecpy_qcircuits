import numbers
from os import times
import traceback
from exopy.tasks.tasks.task_interface import TaskInterface
import numpy as np
from inspect import cleandoc
from math import ceil, log2

from atom.api import Bool, Str, Enum, set_default

from exopy.tasks.api import InstrumentTask, InterfaceableTaskMixin, validators
from numpy.core.records import record

VAL_REAL = validators.Feval(types=numbers.Real)

VAL_INT = validators.Feval(types=numbers.Integral)

class RealTimeAcquisitionTask(InterfaceableTaskMixin, InstrumentTask):
    """ Get the raw or averaged quadratures of the signal.
        Can also get raw or averaged traces of the signal.
        Power demodulation (signal squaring) and custom
        shapes for demodulation are also supported.
    """
    average          = Bool(True).tag(pref=True)
    nof_records      = Str('100000').tag(pref=True, feval=VAL_INT)
    nof_experiments  = Str('1').tag(pref=True,feval=VAL_INT)
    trigger_level    = Str('1.0').tag(pref=True, feval=VAL_REAL)
    enable_aux_trig  = Bool(False).tag(pref=True)
    defer_process    = Bool(False).tag(pref=True)
    sampling_freq    = Str('500').tag(pref=True) # Default for Alazar, overriden by task interface
    timeout          = Str('1000').tag(pref=True)
    trace_start_A    = Str('0').tag(pref=True)
    trace_start_B    = Str('0').tag(pref=True)
    trace_duration_A = Str('0').tag(pref=True)
    trace_duration_B = Str('0').tag(pref=True)
    timesteps_A      = Str('0').tag(pref=True)
    timesteps_B      = Str('0').tag(pref=True)
    demod_start_A    = Str('0').tag(pref=True)
    demod_start_B    = Str('0').tag(pref=True)
    demod_freq_A     = Str('50').tag(pref=True)
    demod_freq_B     = Str('50').tag(pref=True)
    demod_duration_A = Str('0').tag(pref=True)
    demod_duration_B = Str('0').tag(pref=True)
    power_A          = Bool(False).tag(pref=True)
    power_B          = Bool(False).tag(pref=True)
    custom_demod_cos = Str('[]').tag(pref=True)
    custom_demod_sin = Str('[]').tag(pref=True)

    database_entries = set_default({'Trace': {}, 'Demod': {}, 'Power': {}})

    def format_ints(self, string, factor, n, traceback, traceback_root, test):
        """
        """
        ls = self.format_string(string, factor, n)
        for x in ls:
            if abs(x - int(x)) > 1e-6:
                test = False
                traceback[traceback_root+'-not_integral_nsamples'] = cleandoc(f'{string}: Number of samples must be an integer.')
        return test, traceback, list(map(int, ls))

    def format_string(self, string, factor, n):
        """
        """
        s = self.format_and_eval_string(string)
        if isinstance(s, list) or isinstance(s, tuple) or isinstance(s, np.ndarray):
            return [elem*factor for elem in s]
        else:
            return [s*factor]*n

    def parse_inputs(self, test, traceback, traceback_root):
        """
        """
        nof_records      = self.format_and_eval_string(self.nof_records)
        nof_experiments  = max(1, self.format_and_eval_string(self.nof_experiments))
        trigger_level    = self.format_and_eval_string(self.trigger_level)
        sampling_freq    = self.format_and_eval_string(self.sampling_freq) # In MS/s or samples/µs
        timeout          = max(0, self.format_and_eval_string(self.timeout))
        samples_per_ns   = sampling_freq / 1000
        average          = self.average
        enable_aux_trig  = self.enable_aux_trig
        defer_process    = self.defer_process
        power_A          = self.power_A
        power_B          = self.power_B
        power            = [power_A, power_B]
        test, traceback, trace_start_A    = self.format_ints(self.trace_start_A,    samples_per_ns, 1, traceback, traceback_root, test) # ns to samples conversion
        test, traceback, trace_start_B    = self.format_ints(self.trace_start_B,    samples_per_ns, 1, traceback, traceback_root, test) # ns to samples conversion
        test, traceback, trace_duration_A = self.format_ints(self.trace_duration_A, samples_per_ns, 1, traceback, traceback_root, test) # ns to samples conversion
        test, traceback, trace_duration_B = self.format_ints(self.trace_duration_B, samples_per_ns, 1, traceback, traceback_root, test) # ns to samples conversion
        test, traceback, timesteps_A      = self.format_ints(self.timesteps_A,      samples_per_ns, 1, traceback, traceback_root, test) # ns to samples conversion
        test, traceback, timesteps_B      = self.format_ints(self.timesteps_B,      samples_per_ns, 1, traceback, traceback_root, test) # ns to samples conversion
        test, traceback, demod_start_A    = self.format_ints(self.demod_start_A,    samples_per_ns, 1, traceback, traceback_root, test) # ns to samples conversion
        test, traceback, demod_start_B    = self.format_ints(self.demod_start_B,    samples_per_ns, 1, traceback, traceback_root, test) # ns to samples conversion
        test, traceback, demod_duration_A = self.format_ints(self.demod_duration_A, samples_per_ns, 1, traceback, traceback_root, test) # ns to samples conversion
        test, traceback, demod_duration_B = self.format_ints(self.demod_duration_B, samples_per_ns, 1, traceback, traceback_root, test) # ns to samples conversion
        demod_freq_A     = self.format_string(self.demod_freq_A,     1, 1)
        demod_freq_B     = self.format_string(self.demod_freq_B,     1, 1)
        custom_demod_cos = self.format_and_eval_string(self.custom_demod_cos)
        custom_demod_sin = self.format_and_eval_string(self.custom_demod_sin)

        if nof_records % nof_experiments != 0:
            test = False
            traceback[traceback_root+'-npoints_must_divide_ntraces'] = cleandoc('The number of returned points must divide the number of traces.')

        if len(trace_start_A) != len(trace_duration_A):
            test = False
            traceback[traceback_root] = cleandoc('Channel A raw trace record settings "Start time after '
                                                 'trigger (ns)" and "Duration (ns)" must have the same length.')
        if len(trace_start_B) != len(trace_duration_B):
            test = False
            traceback[traceback_root] = cleandoc('Channel B raw trace record settings "Start time after '
                                                 'trigger (ns)" and "Duration (ns)" must have the same length.')
        trace_A = np.array(list(zip(trace_start_A, trace_duration_A)))
        trace_B = np.array(list(zip(trace_start_B, trace_duration_B)))
        trace   = [trace_A, trace_B]

        if not(len(demod_start_A) == len(demod_duration_A) == len(demod_freq_A)):
            test = False
            traceback[traceback_root] = cleandoc('Channel A demodulation settings "Start time after trigger (ns)", '
                                                 '"Duration (ns)" and "Frequency (MHz)" must have the same length.')
        if not(len(demod_start_B) == len(demod_duration_B) == len(demod_freq_B)):
            test = False
            traceback[traceback_root] = cleandoc('Channel B demodulation settings "Start time after trigger (ns)", '
                                                 '"Duration (ns)" and "Frequency (MHz)" must have the same length.')
        demod_A = np.array(list(zip(demod_start_A, demod_duration_A, demod_freq_A)))
        demod_B = np.array(list(zip(demod_start_B, demod_duration_B, demod_freq_B)))
        demod   = [demod_A, demod_B]
        timesteps = [timesteps_A, timesteps_B]
        for c, demod_c in enumerate(demod):
            for start, duration, freq in demod_c:
                if duration < 0:
                    test = False
                    traceback[traceback_root] = cleandoc(f'Channel {"AB"[c]} demodulation setting "Duration" must be positive.')
                    return test, {}
        for c, trace_c in enumerate(trace):
            for start, duration in trace_c:
                if duration < 0:
                    test = False
                    traceback[traceback_root] = cleandoc(f'Channel {"AB"[c]} raw trace record "Duration" must be positive.')
                    return test, {}

        enable_channel_A = all([duration != 0 for start, duration in trace_A]) or all([duration != 0 for start, duration, freq in demod_A])
        enable_channel_B = all([duration != 0 for start, duration in trace_B]) or all([duration != 0 for start, duration, freq in demod_B])
        active_channels  = []
        if enable_channel_A:
            active_channels.append(0)
        if enable_channel_B:
            active_channels.append(1)
        active_channels = np.array(active_channels)
        if any([duration == 0 for start, duration in trace_A]):
            trace_A = []
        if any([duration == 0 for start, duration, freq in demod_A]):
            demod_A     = []
            timesteps_A = []
        if any([duration == 0 for start, duration in trace_B]):
            trace_B = []
        if any([duration == 0 for start, duration, freq in demod_B]):
            demod_B     = []
            timesteps_B = []
        demod     = [demod_A,     demod_B]
        timesteps = [timesteps_A, timesteps_B]

        for ch in active_channels:
            if len(timesteps[ch]) == 1:
                timesteps[ch] *= len(demod[ch])
            for i, (start, duration, freq) in enumerate(demod[ch]):
                if timesteps[ch][i] == 0:
                    timesteps[ch][i] = duration
            for i, ((demod_start, demod_duration, demod_freq), timestep) in enumerate(zip(demod[ch], timesteps[ch])):
                if demod_duration % timestep != 0:
                    test = False
                    traceback[traceback_root] = cleandoc(f'IQ time step {i} ({timestep/samples_per_ns} ns) must divide '
                                                         f'demodulation duration {i} ({demod_duration/samples_per_ns} ns).')

        record_length = [0, 0]
        offset_start  = [0, 0]
        offset_stop   = [0, 0]
        for ch in active_channels:
            offset_start[ch] = 1000000
            offset_stop[ch]  = 0
            for start, duration, freq in demod[ch]:
                offset_start[ch] = min(offset_start[ch], start)
                offset_stop[ch]  = max(offset_stop[ch],  start + duration)
            for start, duration in trace[ch]:
                offset_start[ch] = min(offset_start[ch], start)
                offset_stop[ch]  = max(offset_stop[ch],  start + duration)
            offset_start[ch]  = 8 * (offset_start[ch]//8)
            record_length[ch] = offset_stop[ch] - offset_start[ch]
        for ch in active_channels:
            for i in range(len(demod[ch])):
                demod[ch][i][0] -= offset_start[ch]
            for i in range(len(trace[ch])):
                trace[ch][i][0] -= offset_start[ch]
        record_length = np.array(record_length)
        offset_start  = np.array(offset_start)
        offset_stop   = np.array(offset_stop)

        if any([horiz_offset % 8 != 0 or horiz_offset < -16360 or horiz_offset > 2**35-8 for horiz_offset in offset_start]):
            test = False
            traceback[traceback_root] = cleandoc('Horizontal offset for a record (in samples) must'
                                                 ' be a multiple of 8 between -16360 and 2^35-8.')
            return test, {}

        cos_1      = [[], []]
        sin_1      = [[], []]
        cos_2      = [[], []]
        sin_2      = [[], []]
        bit_shifts = [[], []]
        for c, ch in enumerate(active_channels):
            for i, ((start, duration, freq), timestep) in enumerate(zip(demod[ch], timesteps[ch])):
                period = sampling_freq/freq
                if duration % timestep != 0:
                    test = False
                    traceback[traceback_root+'-demod_time_step_duration'] = cleandoc(f'All demodulation time steps must divide the demodulation'
                                                                                     f' duration. A time step of {timestep/samples_per_ns} ns '
                                                                                     f'is incompatible with a duration of {duration/samples_per_ns} ns.')
                if abs(timestep/period - round(timestep/period)) > 1e-6:
                    test = False
                    traceback[traceback_root+'-demod_time_step_period'] = cleandoc(f'All demodulation periods must divide IQ time steps. A period of'
                                                                                   f' {period/samples_per_ns} ns is incompatible with a time step of {duration/samples_per_ns} ns.')
                period = timestep / round(timestep/period)
                if len(custom_demod_cos) and len(custom_demod_sin):
                    if len(custom_demod_cos) != len(custom_demod_sin):
                        test = False
                        traceback[traceback_root+'-demod_cos_sin_npoints_mismatch'] = cleandoc('Custom demodulation "cos" and "sin" must have the same length.')
                    if len(custom_demod_cos) < duration:
                        test = False
                        traceback[traceback_root+'-demod_function_npoints_mismatch_duration'] = cleandoc(f'Custom demodulation functions ({len(custom_demod_cos)} points each) have fewer values than demodulation duration'
                                                                                                         f' #{i} for channel {"AB"[ch]} ({duration/samples_per_ns} ns = {duration} samples @ {sampling_freq} MS/s).')
                    cos_func = (custom_demod_cos[:duration]*(2**15-1) / np.max(np.abs(custom_demod_cos[:duration]))).tolist()
                    sin_func = (custom_demod_sin[:duration]*(2**15-1) / np.max(np.abs(custom_demod_sin[:duration]))).tolist()
                else:
                    x = np.linspace(0, 2*np.pi*duration/period, num=duration)
                    cos_func = (np.cos(x)*(2**15-1)).tolist()
                    sin_func = (np.sin(x)*(2**15-1)).tolist()
                cos_func = ([0] * start) + cos_func + ([0] * (record_length[ch] - start - duration))
                sin_func = ([0] * start) + sin_func + ([0] * (record_length[ch] - start - duration))
                cos_1[ch].append(np.array(cos_func, dtype=np.int16))
                sin_1[ch].append(np.array(sin_func, dtype=np.int16))
                if power[ch] and defer_process:
                    cos_2[ch].append(np.array(cos_func, dtype=np.float32)**2 * ((2**16-1)/2**30))
                    sin_2[ch].append(np.array(sin_func, dtype=np.float32)**2 * ((2**16-1)/2**30))
                elif power[ch]:
                    cos_2[ch].append(np.uint16(np.array(cos_func, dtype=np.int32)**2 * ((2**16-1)/2**30)))
                    sin_2[ch].append(np.uint16(np.array(sin_func, dtype=np.int32)**2 * ((2**16-1)/2**30)))
                    bit_shift = max(0, ceil(log2(nof_records) + log2(max([timestep for timestep in timesteps[ch]])) + 30 + log2(np.sum(cos_2[ch][-1], axis=-1))) - 64)
                    if bit_shift > 12:
                        test = False
                        traceback[traceback_root+'-demod_func-precision_underflow'] = cleandoc('Overflow (too many points): number of records, '
                                                                                               'demodulation durations, raw trace record durations.')
                        bit_shift = 12
                    if bit_shift > 0:
                        traceback[traceback_root+'-demod_func-decreased_precision'] = cleandoc(f'Demodulation functions use {16-bit_shift} bits'
                                                                                               f' instead of 16 to avoid integer overflows.')
                        cos_2[ch][-1] = cos_2[ch][-1] >> bit_shift
                        sin_2[ch][-1] = sin_2[ch][-1] >> bit_shift
                    bit_shifts[ch].append(bit_shift)

        min_matrix_size = 8_000_000 # Samples
        min_timesteps   = min(timesteps_A + timesteps_B + [duration for start, duration in list(trace_A) + list(trace_B)])
        min_buffer_size = min_matrix_size // min_timesteps
        records_per_buf = nof_experiments * int(ceil(min_buffer_size / nof_experiments))
        records_per_buf = min(records_per_buf, nof_records)
        nof_buffers     = int(ceil(nof_records / records_per_buf))

        return test, {'trace_A':         trace_A,
                      'trace_B':         trace_B,
                      'demod_A':         demod_A,
                      'demod_B':         demod_B,
                      'power_A':         power_A,
                      'power_B':         power_B,
                      'timesteps_A':     timesteps_A,
                      'timesteps_B':     timesteps_B,
                      'cos_1':           cos_1,
                      'sin_1':           sin_1,
                      'cos_2':           cos_2,
                      'sin_2':           sin_2,
                      'sampling_freq':   sampling_freq,
                      'timeout':         timeout,
                      'active_channels': active_channels,
                      'defer_process':   defer_process,
                      'average':         average,
                      'enable_aux_trig': enable_aux_trig,
                      'bit_shifts':      bit_shifts,
                      'nof_records':     nof_records,
                      'records_per_buf': records_per_buf,
                      'record_length':   record_length,
                      'nof_buffers':     nof_buffers,
                      'trigger_level':   trigger_level,
                      'offset_start':    offset_start,
                      'nof_experiments': nof_experiments}

    def check(self, *args, **kwargs):
        """
        """
        test, traceback = super().check(*args, **kwargs)
        traceback_root  = self.path + '/' + self.name + '-get_demod'
        test, ret       = self.parse_inputs(test, traceback, traceback_root)

        return test, traceback

    def i_perform(self):
        """
        """
        test, ret = self.parse_inputs(test = True, traceback = {}, traceback_root = '')

        trace_A         = ret['trace_A']
        trace_B         = ret['trace_B']
        demod_A         = ret['demod_A']
        demod_B         = ret['demod_B']
        power_A         = ret['power_A']
        power_B         = ret['power_B']
        timesteps_A     = ret['timesteps_A']
        timesteps_B     = ret['timesteps_B']
        cos_1           = ret['cos_1']
        sin_1           = ret['sin_1']
        cos_2           = ret['cos_2']
        sin_2           = ret['sin_2']
        sampling_freq   = ret['sampling_freq']
        timeout         = ret['timeout']
        active_channels = ret['active_channels']
        defer_process   = ret['defer_process']
        average         = ret['average']
        enable_aux_trig = ret['enable_aux_trig']
        bit_shifts      = ret['bit_shifts']
        nof_records     = ret['nof_records']
        records_per_buf = ret['records_per_buf']
        record_length   = ret['record_length']
        nof_buffers     = ret['nof_buffers']
        trigger_level   = ret['trigger_level']
        offset_start    = ret['offset_start']
        nof_experiments = ret['nof_experiments']

        self.driver.configure_board(sampling_freq, trigger_level, active_channels, record_length, records_per_buf*nof_buffers, records_per_buf, offset_start)

        tr, dm, pw = self.driver.get_demod_irt(trace_A, trace_B, demod_A, demod_B, power_A, power_B,
                                               timesteps_A, timesteps_B, cos_1, sin_1, cos_2, sin_2,
                                               defer_process, average, nof_experiments, nof_records,
                                               record_length, records_per_buf, enable_aux_trig,
                                               timeout, active_channels, bit_shifts)

        self.write_in_database('Trace', tr)
        self.write_in_database('Demod', dm)
        self.write_in_database('Power', pw)


class TeledyneInterface(TaskInterface):
    sampling_freq = Enum('2500','1250', '625').tag(pref=True)
    
    def check(self, *args, **kwargs):
        self.task.sampling_freq = self.sampling_freq

        return True, {}

    def perform(self, *args, **kwargs):
        self.task.sampling_freq = self.sampling_freq

        return self.task.i_perform(*args, **kwargs)