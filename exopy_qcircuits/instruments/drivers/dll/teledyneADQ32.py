from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

import sys
from subprocess import call
import ctypes
import os
from inspect import cleandoc

import numpy as np
import math
import time
import logging

from numpy.core.fromnumeric import reshape, trace
from numpy.core.records import record

logger = logging.getLogger(__name__)

from ..dll_tools import DllInstrument

import pyadq
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import ctypes as ct
import time
from numba import njit
from math import ceil, log2

from pyadq import RecordBuffer
from pyadq.structs import _ADQP2pStatus, ADQ_EAGAIN
from pyadq.error import ApiError

def WaitForP2pBuffers(self, timeout: int) -> RecordBuffer:
    status = _ADQP2pStatus()

    result = self.ADQ_WaitForP2pBuffers(
        ct.byref(status),
        timeout
    )

    if result == ADQ_EAGAIN:
        raise ApiError("Timed out while waiting for peer-to-peer buffers.")
    elif result < 0:
        raise ApiError(f"Waiting for peer-to-peer markers failed with retcode {result}.")

    return status._to_native()

@njit
def demodulate(record, cosine, sine, I_arr, Q_arr):
    i, j = 0, 0
    while i < len(record): # for i in range(NOF_RECORDS_PER_BUFFER)
        j = 0
        while j < len(record[0]): # for j in range(record_length)
            I_arr[i, 0] += record[i,j]*cosine[j]
            Q_arr[i, 0] += record[i,j]*sine[j]
            j += 1
        i += 1

@njit
def demodulate_tstep(record, cosine, sine, timestep, I_arr, Q_arr):
    i, j, div, mod = 0, 0, 0, 0 # div, mod = j // timestep, j % timestep
    while i < len(record): # for i in range(NOF_RECORDS_PER_BUFFER)
        j, div, mod = 0, 0, 0
        while j < len(record[0]): # for j in range(record_length)
            if mod == timestep:
                div += 1
                mod = 0
            I_arr[i, div] += record[i,j]*cosine[j]
            Q_arr[i, div] += record[i,j]*sine[j]
            j += 1
            mod += 1
        i += 1

@njit
def demodulate_power(record, cosine_sq, sine_sq, I_arr, Q_arr):
    i, j = 0, 0
    while i < len(record): # for i in range(NOF_RECORDS_PER_BUFFER)
        j = 0
        while j < len(record[0]): # for j in range(record_length)
            I_arr[i, 0] += (record[i,j]**2) * cosine_sq[j]
            Q_arr[i, 0] += (record[i,j]**2) * sine_sq[j]
            j += 1
        i += 1

@njit
def demodulate_power_tstep(record, cosine_sq, sine_sq, timestep, I_arr, Q_arr):
    i, j, div, mod = 0, 0, 0, 0 # div, mod = j // timestep, j % timestep
    while i < len(record): # for i in range(NOF_RECORDS_PER_BUFFER)
        j, div, mod = 0, 0, 0
        while j < len(record[0]): # for j in range(record_length)
            if mod == timestep:
                div += 1
                mod = 0
            I_arr[i, div] += (record[i,j]**2) * cosine_sq[j]
            Q_arr[i, div] += (record[i,j]**2) * sine_sq[j]
            j += 1
            mod += 1
        i += 1

class TeledyneADQ32(DllInstrument):
    def __init__(self, connection_infos, caching_allowed=True, caching_permissions={}, auto_open=True):
        super().__init__(connection_infos, caching_allowed, caching_permissions, auto_open)
        if auto_open:
            self.open_connection()

    def open_connection(self):
        self.acu = pyadq.ADQControlUnit()
        self.acu.ADQControlUnit_EnableErrorTrace(pyadq.LOG_LEVEL_INFO, '.')
        device_list      = self.acu.ListDevices()
        device_to_open   = 0
        self.dev         = self.acu.SetupDevice(device_to_open)
        params           = self.dev.GetParameters(pyadq.ADQ_PARAMETER_ID_ANALOG_FRONTEND)
        self.INPUT_RANGE = params.channel[0].input_range
        self.CODE_TO_V   = 1e-3 * self.INPUT_RANGE / 2**16
        self.dev.WaitForP2pBuffers = WaitForP2pBuffers

    def close_connection(self):
        pass

    def configure_board(self, sampling_freq, trigger_level, active_channels, record_length, nof_records_tot, records_per_buf, offset_start):
        """Configure the acquisition board before any acquisition and demodulation.

        Parameters
        ----------
        sampling_freq: int
            Sampling frequency in MS/s or samples/µs.
        trigger_level: float
            Voltage threshold in V.
        active_channels: list
            List of active channels indices (0 for channel A, 1 for channel B).
        record_length: int
            Number of samples per record.
        nof_records_tot: int
            nof_buffers * records_per_buffers.
        records_per_buffer: int
            Number of records in each buffer.
        offset_start:
            Earliest sample among all raw traces and demodulations.
        """
        cur_config = getattr(self, 'config', {})
        new_config = {'sampling_freq':   sampling_freq,
                      'trigger_level':   trigger_level,
                      'active_channels': active_channels,
                      'record_length':   record_length,
                      'nof_records_tot': nof_records_tot,
                      'records_per_buf': records_per_buf,
                      'offset_start':    offset_start}

        if not cur_config or new_config != cur_config:
            self.config  = new_config
            SIZEOF_INT16 = 2

            parameters = self.dev.InitializeParameters(pyadq.ADQ_PARAMETER_ID_TOP)

            parameters.constant.clock_system.low_jitter_mode_enabled = True
            parameters.constant.clock_system.clock_generator         = pyadq.ADQ_CLOCK_GENERATOR_INTERNAL_PLL    # int clk source
            parameters.constant.clock_system.reference_source        = pyadq.ADQ_CLOCK_REFERENCE_SOURCE_PORT_CLK # ext ref

            for ch in active_channels:
                parameters.signal_processing.sample_skip.channel[ch].skip_factor = 2500//sampling_freq

                acq_param                   = parameters.acquisition.channel[ch]
                acq_param.record_length     = record_length[ch]
                acq_param.nof_records       = nof_records_tot
                acq_param.trigger_source    = pyadq.ADQ_EVENT_SOURCE_TRIG
                acq_param.horizontal_offset = offset_start[ch]

            parameters.transfer.common.marker_mode                      = 1 # ADQ_MARKER_MODE_HOST_MANUAL
            parameters.transfer.common.write_lock_enabled               = 1
            parameters.transfer.common.packed_buffers_enabled           = 0
            parameters.transfer.common.transfer_records_to_host_enabled = 1

            for ch in active_channels:
                transfer_param                                = parameters.transfer.channel[ch]
                transfer_param.nof_buffers                    = pyadq.ADQ_MAX_NOF_BUFFERS
                transfer_param.record_size                    = SIZEOF_INT16 * record_length[ch]
                transfer_param.record_length_infinite_enabled = 0
                transfer_param.record_buffer_size             = records_per_buf * (SIZEOF_INT16 * record_length[ch])
                transfer_param.metadata_enabled               = 0

            parameters.event_source.port[0].threshold = trigger_level

            self.dev.SetParameters(parameters)

    def get_demod_irt(self, trace_A, trace_B, demod_A, demod_B, power_A, power_B,
                            tstep_A, tstep_B, cos_1, sin_1, cos_2, sin_2,
                            defer_process, average, nof_experiments, nof_records,
                            record_length, records_per_buf, enable_aux_trig,
                            timeout, active_channels, bit_shifts):
        """Handle raw trace recording and/or voltage/power demodulation.
        
        Parameters
        ----------
        trace_A: list
            List of (start, duration) sequences where "duration" must be strictly positive. All values are given in samples.
        trace_B: list
            List of (start, duration) sequences where "duration" must be strictly positive. All values are given in samples.
        demod_A: list
            List of (start, duration, freq) sequences where "duration" must be strictly positive. All values are given in samples.
        demod_B: list
            List of (start, duration, freq) sequences where "duration" must be strictly positive. All values are given in samples.
        power_A: bool
            If True, power demodulation (from squared signal) is computed.
        power_B: bool
            If True, power demodulation (from squared signal) is computed.
        tstep_A: list
            List of timestep values in samples (one value per element in trace_A).
        tstep_B: list
            List of timestep values in samples (one value per element in trace_B).
        cos_1: list
            List of two lists giving cos (or custom cos) int16 values for each channel (A and B).
        sin_1: list
            List of two lists giving sin (or custom sin) int16 values for each channel (A and B).
        cos_2: list
            List of two lists giving cos² (or custom cos²) uint16 (defer_process == False) or float32 (defer_process == True) values for each channel (A and B).
        sin_2: list
            List of two lists giving cos² (or custom cos²) uint16 (defer_process == False) or float32 (defer_process == True) values for each channel (A and B).
        defer_process: bool
            If True, raw data is saved to be processed after its acquisition.
        average: bool
            Average multiple repetitions of the same experiments.
        nof_experiments: int
            Number of points returned when average == True.
        nof_records:
            Number of individual records.
        record_length: int
            Number of samples per record.
        records_per_buf:
            Number of records per memory buffer.
        enable_aux_trig:
            If True, the card will send a 100ms 5V pulse on its AUX GPIO port whenever it is ready to start capturing data or is done capturing data.
        timeout: int
            Positive timeout in ms passed to WaitForP2pBuffers.
        active_channels: list
            List of active channels indices (0 for channel A, 1 for channel B).
        bit_shifts:
            List of two lists giving cos² and sin² loss of precision (in bits) required to avoid overflows. Only used if defer_process == False.

        Returns
        -------
        tr:
            not average: structured array of raw traces. Format: tr['WXXXX'] where W = A or B (channel) and XXXX is the raw trace number.
            average: structured array of averaged raw traces. Format: tr['WXXXX'] where W = A or B (channel) and XXXX is the raw trace number.
        dm:
            not average: structured array of demodulated data. Format: dm['WXY_Z'] or dm['WXY'] where W = A or B (channel), X = I or Q (phase), Y is the raw trace number and Z the timestep number (if any).
            average: structured array of demodulated data. Format: dm['WXY_Z'] or dm['WXY'] where W = A or B (channel), X = I or Q (phase), Y is the raw trace number and Z the timestep number (if any).
        pw:
            not average: structured array of demodulated power. Format: dm['WX_Y'] or dm['WX'] where W = A or B (channel), X is the raw trace number and Y the timestep number (if any).
            average: structured array of demodulated power. Format: dm['WX_Y'] or dm['WX'] where W = A or B (channel), X is the raw trace number and Y the timestep number (if any).
        """
        nof_buffers = int(ceil(nof_records / records_per_buf))

        N_trace_A = len(trace_A)
        N_trace_B = len(trace_B)
        N_demod_A = len(demod_A)
        N_demod_B = len(demod_B)

        trace = [trace_A, trace_B]
        demod = [demod_A, demod_B]
        power = [power_A, power_B]
        tstep = [tstep_A, tstep_B]

        I_tot    = [[], []]
        Q_tot    = [[], []]
        I_tot_sq = [[], []]
        Q_tot_sq = [[], []]
        raw_data = [[], []]

        trace_start_ch    = [[start    for start, duration       in trace_ch] for trace_ch in trace] # Per channel
        trace_duration_ch = [[duration for start, duration       in trace_ch] for trace_ch in trace] # Per channel
        demod_start_ch    = [[start    for start, duration, freq in demod_ch] for demod_ch in demod] # Per channel
        demod_duration_ch = [[duration for start, duration, freq in demod_ch] for demod_ch in demod] # Per channel
        demod_freq_ch     = [[freq     for start, duration, freq in demod_ch] for demod_ch in demod] # Per channel

        trace_start_fl    = trace_start_ch[0]    + trace_start_ch[1]                                 # Flat
        trace_duration_fl = trace_duration_ch[0] + trace_duration_ch[1]                              # Flat
        demod_start_fl    = demod_start_ch[0]    + demod_start_ch[1]                                 # Flat
        demod_duration_fl = demod_duration_ch[0] + demod_duration_ch[1]                              # Flat
        demod_freq_fl     = demod_freq_ch[0]     + demod_freq_ch[1]                                  # Flat

        tstep_fl = []
        for tstep_ch in tstep:
            tstep_fl += tstep_ch

        for ch in active_channels:
            if average and nof_experiments == 1:
                I_tot[ch]    = [np.zeros(duration // tstep[ch][i], dtype=np.int64)  for i, (start, duration, freq) in enumerate(demod[ch])]
                Q_tot[ch]    = [np.zeros(duration // tstep[ch][i], dtype=np.int64)  for i, (start, duration, freq) in enumerate(demod[ch])]
                I_tot_sq[ch] = [np.zeros(duration // tstep[ch][i], dtype=np.uint64) for i, (start, duration, freq) in enumerate(demod[ch])]
                Q_tot_sq[ch] = [np.zeros(duration // tstep[ch][i], dtype=np.uint64) for i, (start, duration, freq) in enumerate(demod[ch])]
                raw_data[ch] = [np.zeros((duration,), dtype=np.int64) for start, duration in trace[ch]]
            elif average:
                I_tot[ch]    = [np.zeros((nof_experiments, duration // tstep[ch][i]), dtype=np.int64)  for i, (start, duration, freq) in enumerate(demod[ch])]
                Q_tot[ch]    = [np.zeros((nof_experiments, duration // tstep[ch][i]), dtype=np.int64)  for i, (start, duration, freq) in enumerate(demod[ch])]
                I_tot_sq[ch] = [np.zeros((nof_experiments, duration // tstep[ch][i]), dtype=np.uint64) for i, (start, duration, freq) in enumerate(demod[ch])]
                Q_tot_sq[ch] = [np.zeros((nof_experiments, duration // tstep[ch][i]), dtype=np.uint64) for i, (start, duration, freq) in enumerate(demod[ch])]
                raw_data[ch] = [np.zeros((nof_experiments, duration), dtype=np.int64) for start, duration in trace[ch]]
            else:
                I_tot[ch]    = [np.empty((nof_records, duration // tstep[ch][i]), dtype=np.int64)  for i, (start, duration, freq) in enumerate(demod[ch])]
                Q_tot[ch]    = [np.empty((nof_records, duration // tstep[ch][i]), dtype=np.int64)  for i, (start, duration, freq) in enumerate(demod[ch])]
                I_tot_sq[ch] = [np.empty((nof_records, duration // tstep[ch][i]), dtype=np.uint64) for i, (start, duration, freq) in enumerate(demod[ch])]
                Q_tot_sq[ch] = [np.empty((nof_records, duration // tstep[ch][i]), dtype=np.uint64) for i, (start, duration, freq) in enumerate(demod[ch])]
                raw_data[ch] = [np.empty((nof_records, duration), dtype=np.int16) for start, duration in trace[ch]]

        compiled = False

        if len(demod_A) or len(demod_B):
            I_arr = [[], []]
            Q_arr = [[], []]
            I_sq  = [[], []]
            Q_sq  = [[], []]

            for ch in active_channels:
                if len(demod[ch]):
                    I_arr[ch] = [np.zeros((records_per_buf, duration // tstep[ch][i]), dtype=np.int64)  for i, (start, duration, freq) in enumerate(demod[ch])]
                    Q_arr[ch] = [np.zeros((records_per_buf, duration // tstep[ch][i]), dtype=np.int64)  for i, (start, duration, freq) in enumerate(demod[ch])]
                    I_sq[ch]  = [np.zeros((records_per_buf, duration // tstep[ch][i]), dtype=np.uint64) for i, (start, duration, freq) in enumerate(demod[ch])]
                    Q_sq[ch]  = [np.zeros((records_per_buf, duration // tstep[ch][i]), dtype=np.uint64) for i, (start, duration, freq) in enumerate(demod[ch])]

                    if not compiled:
                        compiled = True
                        demodulate(      np.zeros((records_per_buf, record_length[0]), dtype=np.int16), cos_1[ch][0], sin_1[ch][0],              I_arr[ch][0], Q_arr[ch][0])
                        demodulate_tstep(np.zeros((records_per_buf, record_length[0]), dtype=np.int16), cos_1[ch][0], sin_1[ch][0], tstep[0][0], I_arr[ch][0], Q_arr[ch][0])
                        if power[ch]:
                            demodulate_power(      np.zeros((records_per_buf, record_length[0]), dtype=np.int16), cos_2[ch][0], sin_2[ch][0],              I_sq[ch][0],  Q_sq[ch][0])
                            demodulate_power_tstep(np.zeros((records_per_buf, record_length[0]), dtype=np.int16), cos_2[ch][0], sin_2[ch][0], tstep[0][0], I_sq[ch][0],  Q_sq[ch][0])

        if defer_process:
            data_tot   = [np.empty((nof_buffers, records_per_buf, record_length[ch]), dtype=np.int16) for ch in (0, 1)]
            buf_params = [np.empty((nof_buffers, 4), dtype=np.uint32) for ch in (0, 1)]

        nof_buffers_received = [0, 0] # Channel A, channel B

        # Start the data acquisition
        result = self.dev.ADQ_StartDataAcquisition()
        if result != pyadq.ADQ_EOK:
            raise Exception(f"ADQ_StartDataAcquisition failed with error code {result}. See log file.")
        print("Starting data acquisition")

        # 100ms aux trig pulse
        if enable_aux_trig:
            aux_trig_parameters                  = self.dev.GetParameters(pyadq.ADQ_PARAMETER_ID_PORT_GPIOA)
            aux_trig_parameters.pin[0].function  = pyadq.ADQ_FUNCTION_GPIO
            aux_trig_parameters.pin[0].direction = pyadq.ADQ_DIRECTION_OUT
            aux_trig_parameters.pin[0].value     = 1
            self.dev.SetParameters(aux_trig_parameters)
            time.sleep(0.1)
            aux_trig_parameters.pin[0].value = 0
            self.dev.SetParameters(aux_trig_parameters)

        transfer_parameters = self.dev.GetParameters(pyadq.ADQ_PARAMETER_ID_DATA_TRANSFER)

        try:
            while any([nof_buffers_received[ch] < ceil(nof_buffers) for ch in active_channels]):
                status = self.dev.WaitForP2pBuffers(self.dev, timeout)

                # Process received buffers
                buf = 0
                while buf < status.channel[0].nof_completed_buffers or buf < status.channel[1].nof_completed_buffers:
                    for ch in active_channels:
                        buffer_index = status.channel[ch].completed_buffers[buf]

                        contents = ct.cast(
                            transfer_parameters.channel[ch].record_buffer[buffer_index], ct.POINTER(ct.c_int16 * (records_per_buf * record_length[ch]))
                        ).contents

                        data = np.frombuffer(contents, dtype=np.int16).reshape((-1, record_length[ch]))

                        start = records_per_buf *  nof_buffers_received[ch]
                        stop  = records_per_buf * (nof_buffers_received[ch]+1)
                        stop2 = records_per_buf

                        # Handle last buffer
                        if nof_records // records_per_buf == nof_buffers_received[ch]:
                            stop2 = nof_records % records_per_buf
                        
                        records_per_experiment = stop2 // nof_experiments

                        # Save raw data to process later
                        if defer_process:
                            data_tot[ch][nof_buffers_received[ch]]     = data
                            buf_params[ch][nof_buffers_received[ch],:] = np.array([start, stop, stop2, records_per_experiment])
                        # Process in real time
                        else:
                            # Save raw data according to `trace`
                            for i, (trace_start, trace_duration) in enumerate(trace[ch]):
                                all_records = data[:stop2,trace_start:trace_start+trace_duration]
                                if average and nof_experiments == 1:
                                    raw_data[ch][i] += np.sum(all_records, axis=0)
                                elif average:
                                    raw_data[ch][i] += np.sum(all_records.reshape((records_per_experiment, nof_experiments, -1)), axis=0)
                                else:
                                    raw_data[ch][i][start:stop] = all_records

                            # Demodulate according to `demod[ch]`
                            for i, (demod_start, demod_duration, demod_freq) in enumerate(demod[ch]):
                                I_arr[ch][i][:] = 0
                                Q_arr[ch][i][:] = 0
                                I_param = I_arr[ch][i][:stop2] # Handle last buffer
                                Q_param = Q_arr[ch][i][:stop2] # Handle last buffer
                                if power[ch]:
                                    I_sq[ch][i][:] = 0
                                    Q_sq[ch][i][:] = 0
                                    I_param_sq = I_sq[ch][i][:stop2]
                                    Q_param_sq = Q_sq[ch][i][:stop2]
                                    if tstep[ch][i] == demod_duration:
                                        demodulate_power(data[:stop2, demod_start:demod_start+demod_duration],
                                                         cos_2[ch][i][demod_start:demod_start+demod_duration],
                                                         sin_2[ch][i][demod_start:demod_start+demod_duration],
                                                         I_param_sq, Q_param_sq)
                                    else:
                                        demodulate_power_tstep(data[:stop2, demod_start:demod_start+demod_duration],
                                                               cos_2[ch][i][demod_start:demod_start+demod_duration],
                                                               sin_2[ch][i][demod_start:demod_start+demod_duration],
                                                               tstep[ch][i],
                                                               I_param_sq, Q_param_sq)
                                if tstep[ch][i] == demod_duration:
                                    demodulate(data[:stop2, demod_start:demod_start+demod_duration],
                                               cos_1[ch][i][demod_start:demod_start+demod_duration],
                                               sin_1[ch][i][demod_start:demod_start+demod_duration],
                                               I_param, Q_param)
                                else:
                                    demodulate_tstep(data[:stop2, demod_start:demod_start+demod_duration],
                                                     cos_1[ch][i][demod_start:demod_start+demod_duration],
                                                     sin_1[ch][i][demod_start:demod_start+demod_duration],
                                                     tstep[ch][i],
                                                     I_param, Q_param)
                                if average and nof_experiments == 1:
                                    I_tot[ch][i] += np.sum(I_param, axis=0)
                                    Q_tot[ch][i] += np.sum(Q_param, axis=0)
                                    if power[ch]:
                                        I_tot_sq[ch][i] += np.sum(I_param_sq, axis=0)
                                        Q_tot_sq[ch][i] += np.sum(Q_param_sq, axis=0)
                                elif average:
                                    I_tot[ch][i] += np.sum(I_param.reshape((records_per_experiment, nof_experiments, -1)), axis=0)
                                    Q_tot[ch][i] += np.sum(Q_param.reshape((records_per_experiment, nof_experiments, -1)), axis=0)
                                    if power[ch]:
                                        I_tot_sq[ch][i] += np.sum(I_param_sq.reshape((records_per_experiment, nof_experiments, -1)), axis=0)
                                        Q_tot_sq[ch][i] += np.sum(Q_param_sq.reshape((records_per_experiment, nof_experiments, -1)), axis=0)
                                else:
                                    I_tot[ch][i][start:stop] = I_param
                                    Q_tot[ch][i][start:stop] = Q_param
                                    if power[ch]:
                                        I_tot_sq[ch][i][start:stop] = I_param_sq
                                        Q_tot_sq[ch][i][start:stop] = Q_param_sq

                        self.dev.ADQ_UnlockP2pBuffers(ch, (1 << buffer_index))

                        nof_buffers_received[ch] += 1
                    buf += 1

        except Exception as e:
            self.dev.ADQ_StopDataAcquisition()
            raise e

        # Stop the data acquisition
        print("Stopping data acquisition")
        result = self.dev.ADQ_StopDataAcquisition()
        if result not in [pyadq.ADQ_EOK, pyadq.ADQ_EINTERRUPTED]:
            raise Exception(f"ADQ_StartDataAcquisition failed with error code {result}. See log file.")

        # 100ms aux trig pulse
        if enable_aux_trig:
            aux_trig_parameters.pin[0].value = 1
            self.dev.SetParameters(aux_trig_parameters)
            time.sleep(0.1)
            aux_trig_parameters.pin[0].value = 0
            self.dev.SetParameters(aux_trig_parameters)

        ############################################################################################

        if defer_process:
            for ch in active_channels:
                for b in range(len(data_tot[ch])):
                    data, (start, stop, stop2, records_per_experiment) = data_tot[ch][b], buf_params[ch][b]
                    for i, (trace_start, trace_duration) in enumerate(trace[ch]):
                        all_records = data[:stop2,trace_start:trace_start+trace_duration]
                        if average and nof_experiments == 1:
                            raw_data[ch][i] += np.sum(all_records, axis=0)
                        elif average:
                            raw_data[ch][i] += np.sum(all_records.reshape((records_per_experiment, nof_experiments, -1)), axis=0)
                        else:
                            raw_data[ch][i][start:stop] = all_records

                    # Demodulate according to `demod[ch]`
                    for i, (demod_start, demod_duration, demod_freq) in enumerate(demod[ch]):
                        I_arr[ch][i][:] = 0
                        Q_arr[ch][i][:] = 0
                        I_param = I_arr[ch][i][:stop2] # Handle last buffer
                        Q_param = Q_arr[ch][i][:stop2] # Handle last buffer
                        if power[ch]:
                            I_sq[ch][i][:] = 0
                            Q_sq[ch][i][:] = 0
                            I_param_sq = I_sq[ch][i][:stop2]
                            Q_param_sq = Q_sq[ch][i][:stop2]
                            if tstep[ch][i] == demod_duration:
                                demodulate_power(data[:stop2, demod_start:demod_start+demod_duration],
                                                 cos_2[ch][i][demod_start:demod_start+demod_duration],
                                                 sin_2[ch][i][demod_start:demod_start+demod_duration],
                                                 I_param_sq, Q_param_sq)
                            else:
                                demodulate_power_tstep(data[:stop2, demod_start:demod_start+demod_duration],
                                                       cos_2[ch][i][demod_start:demod_start+demod_duration],
                                                       sin_2[ch][i][demod_start:demod_start+demod_duration],
                                                        tstep[ch][i],
                                                       I_param_sq, Q_param_sq)
                        if tstep[ch][i] == demod_duration:
                            demodulate(data[:stop2, demod_start:demod_start+demod_duration],
                                       cos_1[ch][i][demod_start:demod_start+demod_duration],
                                       sin_1[ch][i][demod_start:demod_start+demod_duration],
                                       I_param, Q_param)
                        else:
                            demodulate_tstep(data[:stop2, demod_start:demod_start+demod_duration],
                                             cos_1[ch][i][demod_start:demod_start+demod_duration],
                                             sin_1[ch][i][demod_start:demod_start+demod_duration],
                                             tstep[ch][i],
                                             I_param, Q_param)
                        if average and nof_experiments == 1:
                            I_tot[ch][i] += np.sum(I_param, axis=0)
                            Q_tot[ch][i] += np.sum(Q_param, axis=0)
                            if power[ch]:
                                I_tot_sq[ch][i] += np.sum(I_param_sq, axis=0)
                                Q_tot_sq[ch][i] += np.sum(Q_param_sq, axis=0)
                        elif average:
                            I_tot[ch][i] += np.sum(I_param.reshape((records_per_experiment, nof_experiments, -1)), axis=0)
                            Q_tot[ch][i] += np.sum(Q_param.reshape((records_per_experiment, nof_experiments, -1)), axis=0)
                            if power[ch]:
                                I_tot_sq[ch][i] += np.sum(I_param_sq.reshape((records_per_experiment, nof_experiments, -1)), axis=0)
                                Q_tot_sq[ch][i] += np.sum(Q_param_sq.reshape((records_per_experiment, nof_experiments, -1)), axis=0)
                        else:
                            I_tot[ch][i][start:stop] = I_param
                            Q_tot[ch][i][start:stop] = Q_param
                            if power[ch]:
                                I_tot_sq[ch][i][start:stop] = I_param_sq
                                Q_tot_sq[ch][i][start:stop] = Q_param_sq

        ############################################################################################

        print("Normalizing and converting")
        IQ_tot    = [I_tot,    Q_tot]
        IQ_tot_sq = [I_tot_sq, Q_tot_sq]
        for ch in active_channels:
            for i, (demod_start, demod_duration, demod_freq) in enumerate(demod[ch]):
                for iq in (0, 1):
                    if average:
                        IQ_tot[iq][ch][i] = 2 * self.CODE_TO_V * np.float32(IQ_tot[iq][ch][i]) / tstep[ch][i] / nof_records / 2**15
                        if power[ch] and defer_process:
                            IQ_tot_sq[iq][ch][i] = 4 * self.CODE_TO_V**2 * np.float32(IQ_tot_sq[iq][ch][i]) / tstep[ch][i] / nof_records / (2**15)**2
                        elif power[ch]:
                            IQ_tot_sq[iq][ch][i] = 4 * self.CODE_TO_V**2 * np.float32(IQ_tot_sq[iq][ch][i]) / tstep[ch][i] / nof_records / (2**15)**2 * 2**bit_shifts[ch][i]
                    else:
                        IQ_tot[iq][ch][i] = 2 * self.CODE_TO_V * np.float32(IQ_tot[iq][ch][i]) / tstep[ch][i] / 2**15
                        if power[ch] and defer_process:
                            IQ_tot_sq[iq][ch][i] = 4 * self.CODE_TO_V**2 * np.float32(IQ_tot_sq[iq][ch][i]) / tstep[ch][i] / (2**15)**2
                        elif power[ch]:
                            IQ_tot_sq[iq][ch][i] = 4 * self.CODE_TO_V**2 * np.float32(IQ_tot_sq[iq][ch][i]) / tstep[ch][i] / (2**15)**2 * 2**bit_shifts[ch][i]
            for i, (trace_start, trace_duration) in enumerate(trace[ch]):
                raw_data[ch][i] = np.float32(raw_data[ch][i])
                if average:
                    raw_data[ch][i] *= self.CODE_TO_V/nof_records
                else:
                    raw_data[ch][i] *= self.CODE_TO_V
        
        I_tot,    Q_tot    = IQ_tot
        I_tot_sq, Q_tot_sq = IQ_tot_sq
        max_trace_duration = 0
        type_trace         = 'f'
        type_demod         = 'f'
        type_power         = 'f'

        if len(demod_A) or len(demod_B):
            type_demod = []
            type_power = []
            nof_steps  = [int(demod_duration_fl[i]/tstep_fl[i] if tstep_fl[i] else 1) for i in range(N_demod_A + N_demod_B)]
            for i in range(N_demod_A + N_demod_B):
                if i < N_demod_A:
                    char, j = 'A', i
                    index = str(j).zfill(1 + int(np.floor(np.log10(N_demod_A))))
                else:
                    char, j = 'B', i - N_demod_A
                    index = str(j).zfill(1 + int(np.floor(np.log10(N_demod_B))))
                zeros_step = 1 + int(np.floor(np.log10(nof_steps[i])))
                for t in range(nof_steps[i]):
                    iindex = index
                    if nof_steps[i] > 1:
                        iindex = index + '_' + str(t).zfill(zeros_step)
                    type_demod += [(char + 'I' + iindex, 'f'),
                                   (char + 'Q' + iindex, 'f')]
                    type_power += [(char    +    iindex, 'f')]

        if len(trace_A) or len(trace_B):
            zeros_trace_A = 1 + int(np.floor(np.log10(N_trace_A))) if N_trace_A else 0
            zeros_trace_B = 1 + int(np.floor(np.log10(N_trace_B))) if N_trace_B else 0
            type_trace    = ([('A' + str(i).zfill(zeros_trace_A), 'f') for i in range(N_trace_A)]
                          +  [('B' + str(i).zfill(zeros_trace_B), 'f') for i in range(N_trace_B)])
            max_trace_duration = np.max([duration for start, duration in [*trace_A, *trace_B]])

        if average and nof_experiments == 1:
            tr = np.zeros(max_trace_duration, dtype=type_trace)
            dm = np.zeros(1, dtype=type_demod)
            pw = np.zeros(1, dtype=type_power)
        elif average:
            tr = np.zeros((max_trace_duration, nof_experiments), dtype=type_trace)
            dm = np.zeros((1, nof_experiments), dtype=type_demod)
            pw = np.zeros((1, nof_experiments), dtype=type_power)
        else:
            tr = np.zeros((nof_records, max_trace_duration), dtype=type_trace)
            dm = np.zeros( nof_records, dtype=type_demod)
            pw = np.zeros( nof_records, dtype=type_power)

        for i in np.arange(N_demod_A + N_demod_B):
            ch = (i >= N_demod_A)
            if ch == 0:
                dem = i
                char = 'A'
                index = str(dem).zfill(1 + int(np.floor(np.log10(N_demod_A))))
            else:
                dem = i - N_demod_A
                char = 'B'
                index = str(dem).zfill(1 + int(np.floor(np.log10(N_demod_A))))
            zeros_step = 1 + int(np.floor(np.log10(nof_steps[i])))
            for tstp in range(nof_steps[i]):
                iindex = index
                if nof_steps[i] > 1:
                    iindex = index + '_' + str(tstp).zfill(zeros_step)
                if average and nof_experiments == 1:
                    dm[char + 'I' + iindex] = I_tot[ch][dem][tstp]
                    dm[char + 'Q' + iindex] = Q_tot[ch][dem][tstp]
                    if power[ch]:
                        pw[char + iindex] = I_tot_sq[ch][dem][tstp] + Q_tot_sq[ch][dem][tstp]
                elif average:
                    dm[char + 'I' + iindex] = I_tot[ch][dem][:,tstp]
                    dm[char + 'Q' + iindex] = Q_tot[ch][dem][:,tstp]
                    if power[ch]:
                        pw[char + iindex] = I_tot_sq[ch][dem][tstp] + Q_tot_sq[ch][dem][tstp]
                else:
                    dm[char + 'I' + iindex] = np.array(I_tot[ch][dem])[:,tstp]
                    dm[char + 'Q' + iindex] = np.array(Q_tot[ch][dem])[:,tstp]
                    if power[ch]:
                        pw[char + iindex] = np.array(I_tot_sq[ch][dem])[:,tstp] + np.array(Q_tot_sq[ch][dem])[:,tstp]

        for i in np.arange(N_trace_A + N_trace_B):
            k = i + N_demod_A + N_demod_B
            ch = (i >= N_trace_A)

            if ch == 0:
                curr_trace = i
                trace_id = 'A' + str(curr_trace).zfill(zeros_trace_A)
            else:
                curr_trace = i - N_trace_A
                trace_id = 'B' + str(curr_trace).zfill(zeros_trace_B)

            if average and nof_experiments == 1:
                # shape: (max_trace_duration)
                tr[trace_id][:trace_duration_fl[i]] = raw_data[ch][curr_trace]
            elif average:
                # shape: (max_trace_duration, nof_experiments)
                tr[trace_id][:trace_duration_fl[i],:] = raw_data[ch][curr_trace].T
            else:
                # shape: (nof_records, max_trace_duration)
                tr[trace_id][:,:trace_duration_fl[i]] = raw_data[ch][curr_trace]

        return tr, dm, pw