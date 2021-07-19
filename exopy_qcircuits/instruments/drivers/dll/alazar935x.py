# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2017 by exopyQcircuitsLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""This module defines drivers for Alazar using DLL Library.

:Contains:
    Alazar935x

Visual C++ runtime needs to be installed to be able to load the dll.

"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

import sys
from subprocess import call
import ctypes
import os
from inspect import cleandoc

import numpy as np
from numba import njit
import math
import time
import logging

logger = logging.getLogger(__name__)

from ..dll_tools import DllInstrument

try:
    from . import atsapi as ats
except (FileNotFoundError, OSError):
    logger.info("Couldn't find the Alazar DLL, please install the driver "
                "if you want to use it.")


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

class DMABuffer:
    '''Buffer suitable for DMA transfers.

    AlazarTech digitizers use direct memory access (DMA) to transfer
    data from digitizers to the computer's main memory. This class
    abstracts a memory buffer on the host, and ensures that all the
    requirements for DMA transfers are met.

    DMABuffers export a 'buffer' member, which is a NumPy array view
    of the underlying memory buffer

    Args:

      bytes_per_sample (int): The number of bytes per samples of the
      data. This varies with digitizer models and configurations.

      size_bytes (int): The size of the buffer to allocate, in bytes.

    '''
    def __init__(self, bytes_per_sample, size_bytes):
        self.size_bytes = size_bytes
        ctypes.cSampleType = ctypes.c_uint8
        npSampleType = np.uint8
        if bytes_per_sample > 1:
            ctypes.cSampleType = ctypes.c_uint16
            npSampleType = np.uint16

        self.addr = None
        if os.name == 'nt':
            MEM_COMMIT = 0x1000
            PAGE_READWRITE = 0x4
            ctypes.windll.kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_long,
                                                     ctypes.c_long, ctypes.c_long]
            ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
            self.addr = ctypes.windll.kernel32.VirtualAlloc(
                0, ctypes.c_long(size_bytes), MEM_COMMIT, PAGE_READWRITE)
        elif os.name == 'posix':
            ctypes.libc.valloc.argtypes = [ctypes.c_long]
            ctypes.libc.valloc.restype = ctypes.c_void_p
            self.addr = ctypes.libc.valloc(size_bytes)
        else:
            raise Exception("Unsupported OS")

        ctypes.ctypes_array = (ctypes.cSampleType *
                        (size_bytes // bytes_per_sample)
                        ).from_address(self.addr)
        self.buffer = np.frombuffer(ctypes.ctypes_array, dtype=npSampleType)
        pointer, read_only_flag = self.buffer.__array_interface__['data']

    def __exit__(self):
        if os.name == 'nt':
            MEM_RELEASE = 0x8000
            ctypes.windll.kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_long]
            ctypes.windll.kernel32.VirtualFree.restype = ctypes.c_int
            ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.addr), 0, MEM_RELEASE)
        elif os.name == 'posix':
            ctypes.libc.free(self.addr)
        else:
            raise Exception("Unsupported OS")


class Alazar935x(DllInstrument):

    library = 'ATSApi.dll'

    def __init__(self, connection_infos, caching_allowed=True,
                 caching_permissions={}, auto_open=True):

        super(Alazar935x, self).__init__(connection_infos, caching_allowed,
                                         caching_permissions, auto_open)

        ### JEREMY
        self.clock_set = False
        ###

        if auto_open:
            self.open_connection()

    def open_connection(self):
        """Close Alazar app and create the underlying driver.

        """
        try:
            if sys.platform == 'win32':
                call("TASKKILL /F /IM AlazarDSO.exe", shell=True)
        except Exception:
            pass
        self.board = ats.Board()

    def close_connection(self):
        """Do not need to close a connection

        """
        pass

    def configure_board(self,trigRange,trigLevel):
        board = self.board
        # TODO: Select clock parameters as required to generate this
        # sample rate
        samplesPerSec = 500000000.0

        ## JEREMY
        if not self.clock_set:
            board.setCaptureClock(ats.EXTERNAL_CLOCK_10MHz_REF,
                                  500000000,
                                  ats.CLOCK_EDGE_RISING,
                                  0)
            self.clock_set = True

            #board.setCaptureClock(ats.EXTERNAL_CLOCK_10MHz_REF,
            #                          500000000,
            #                          ats.CLOCK_EDGE_RISING,
            #                          0)
            ###

            # TODO: Select channel A input parameters as required.
            board.inputControl(ats.CHANNEL_A,
                                   ats.DC_COUPLING,
                                   ats.INPUT_RANGE_PM_400_MV,
                                   ats.IMPEDANCE_50_OHM)

            # TODO: Select channel A bandwidth limit as required.
            board.setBWLimit(ats.CHANNEL_A, 0)


            # TODO: Select channel B input parameters as required.
            board.inputControl(ats.CHANNEL_B,
                                   ats.DC_COUPLING,
                                   ats.INPUT_RANGE_PM_400_MV,
                                   ats.IMPEDANCE_50_OHM)

            # TODO: Select channel B bandwidth limit as required.
            board.setBWLimit(ats.CHANNEL_B, 0)
            # TODO: Select trigger inputs and levels as required.
            trigCode = int(128 + 127 * trigLevel / trigRange)
            board.setTriggerOperation(ats.TRIG_ENGINE_OP_J,
                                          ats.TRIG_ENGINE_J,
                                          ats.TRIG_EXTERNAL,
                                          ats.TRIGGER_SLOPE_POSITIVE,
                                          trigCode,
                                          ats.TRIG_ENGINE_K,
                                          ats.TRIG_DISABLE,
                                          ats.TRIGGER_SLOPE_POSITIVE,
                                          128)

            # TODO: Select external trigger parameters as required.
            if trigRange == 5:
                board.setExternalTrigger(ats.DC_COUPLING,
                                         ats.ETR_5V)
            else:
                board.setExternalTrigger(ats.DC_COUPLING,
                                         ats.ETR_2V5)

            # TODO: Set trigger delay as required.
            triggerDelay_sec = 0.
            triggerDelay_samples = int(triggerDelay_sec * samplesPerSec + 0.5)
            board.setTriggerDelay(triggerDelay_samples)

            # TODO: Set trigger timeout as required.
            #
            # NOTE: The board will wait for a for this amount of time for a
            # trigger event.  If a trigger event does not arrive, then the
            # board will automatically trigger. Set the trigger timeout value
            # to 0 to force the board to wait forever for a trigger event.
            #
            # IMPORTANT: The trigger timeout value should be set to zero after
            # appropriate trigger parameters have been determined, otherwise
            # the board may trigger if the timeout interval expires before a
            # hardware trigger event arrives.
            board.setTriggerTimeOut(0)
            # Configure AUX I/O connector as required
            board.configureAuxIO(ats.AUX_OUT_TRIGGER,
                                     0)

    def get_demod(self, startaftertrig, duration, recordsPerCapture,
                  recordsPerBuffer, timestep, freq, average, NdemodA, NdemodB,
                  NtraceA, NtraceB, Npoints, demodFormFile, demodCosinus, aux_trig,
				  power):
        board = ats.Board()

        # Number of samples per record: must be divisible by 32
        samplesPerSec = 500000000.0
        samplesPerTrace = int(samplesPerSec * np.max(np.array(startaftertrig) + np.array(duration)))
        if samplesPerTrace % 32 == 0:
            samplesPerRecord = int(samplesPerTrace)
        else:
            samplesPerRecord = int((samplesPerTrace)/32 + 1)*32

        # Compute the number of bytes per record and per buffer
        channel_number = 2 if ((NdemodA or NtraceA) and (NdemodB or NtraceB)) else 1  # Acquisition on A and B
        memorySize_samples, bitsPerSample = board.getChannelInfo()
        bytesPerSample = (bitsPerSample.value + 7) // 8
        bytesPerRecord = bytesPerSample * samplesPerRecord
        bytesPerBuffer = int(bytesPerRecord * recordsPerBuffer*channel_number)

        # For converting data into volts
        channelRange = 0.4 # Volts
        bitsPerSample = 12
        bitShift = 4
        code = (1 << (bitsPerSample - 1)) - 0.5

        bufferCount = int(round(recordsPerCapture / recordsPerBuffer))
        buffers = []
        for i in range(bufferCount):
            buffers.append(DMABuffer(bytesPerSample, bytesPerBuffer))

        # Set the record size
        board.setRecordSize(0, samplesPerRecord)

        # Configure the number of records in the acquisition
        acquisition_timeout_sec = 10
        board.setRecordCount(int(recordsPerCapture))

        # Calculate the number of buffers in the acquisition
        buffersPerAcquisition = round(recordsPerCapture / recordsPerBuffer)

        channelSelect = 1 if not (NdemodB or NtraceB) else (2 if not (NdemodA or NtraceA) else 3)
        board.beforeAsyncRead(channelSelect,  # Channels A & B
                                  0,
                                  samplesPerRecord,
                                  int(recordsPerBuffer),
                                  int(recordsPerCapture),
                                  ats.ADMA_EXTERNAL_STARTCAPTURE |
                                  ats.ADMA_NPT)

        # Post DMA buffers to board. ATTENTION it is very important not to do "for buffer in buffers"
        for i in range(bufferCount):
            buffer = buffers[i]
            board.postAsyncBuffer(buffer.addr, buffer.size_bytes)

        start = time.perf_counter()  # Keep track of when acquisition started
        board.startCapture()  # Start the acquisition

        if aux_trig:
            time.sleep(1)
            board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 1)
            time.sleep(0.1)
            board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 0)

        if time.perf_counter() - start > acquisition_timeout_sec:
            board.abortCapture()
            raise Exception("Error: Capture timeout. Verify trigger")
            time.sleep(10e-3)

        # Preparation of the tables for the demodulation

        startSample = []
        samplesPerDemod = []
        samplesPerBlock = []
        NumberOfBlocks = []
        samplesMissing = []
        data = []
        dataExtended = []

        for i in range(NdemodA + NdemodB):
            startSample.append(int(samplesPerSec * startaftertrig[i]) )
            samplesPerDemod.append(int(samplesPerSec * duration[i]) )

            if timestep[i]:
                samplesPerBlock.append(samplesPerDemod[i] )

            elif not demodCosinus:
                samplesPerBlock.append(samplesPerDemod[i] )

            else:
                # Check wheter it is possible to cut each record in blocks of size equal
                # to an integer number of periods
                periodsPerBlock = 1
                while (periodsPerBlock * samplesPerSec < freq[i] * samplesPerDemod[i]
                       and periodsPerBlock * samplesPerSec % freq[i]):
                    periodsPerBlock += 1
                samplesPerBlock.append(int(np.minimum(periodsPerBlock * samplesPerSec / freq[i],
                                                      samplesPerDemod[i])) )

            NumberOfBlocks.append(np.divide(samplesPerDemod[i], samplesPerBlock[i]) )
            samplesMissing.append((-samplesPerDemod[i]) % samplesPerBlock[i] )
            # Makes the table that will contain the data
            data.append(np.empty((recordsPerCapture, samplesPerBlock[i])) )
            dataExtended.append(np.zeros((recordsPerBuffer, samplesPerDemod[i] + samplesMissing[i]),
                                          dtype='uint16') )

        for i in (np.arange(NtraceA + NtraceB) + NdemodA + NdemodB):
            startSample.append(int(samplesPerSec * startaftertrig[i]) )
            samplesPerDemod.append(int(samplesPerSec * duration[i]) )
            data.append(np.empty((recordsPerCapture, samplesPerDemod[i])) )

        start = time.perf_counter()

        buffersCompleted = 0
        while buffersCompleted < buffersPerAcquisition:

            # Wait for the buffer at the head of the list of available
            # buffers to be filled by the board.
            buffer = buffers[buffersCompleted % len(buffers)]
            board.waitAsyncBufferComplete(buffer.addr, 10000)

            # Process data

            dataRaw = np.reshape(buffer.buffer, (recordsPerBuffer*channel_number, -1))
            dataRaw = dataRaw >> bitShift

            for i in np.arange(NdemodA):
                dataExtended[i][:,:samplesPerDemod[i]] = dataRaw[:recordsPerBuffer,startSample[i]:startSample[i]+samplesPerDemod[i]]
                dataBlock = np.reshape(dataExtended[i],(recordsPerBuffer,-1,samplesPerBlock[i]))
                data[i][buffersCompleted*recordsPerBuffer:(buffersCompleted+1)*recordsPerBuffer] = np.sum(dataBlock, axis=1)

            for i in (np.arange(NdemodB) + NdemodA):
                dataExtended[i][:,:samplesPerDemod[i]] = dataRaw[(channel_number-1)*recordsPerBuffer:channel_number*recordsPerBuffer,startSample[i]:startSample[i]+samplesPerDemod[i]]
                dataBlock = np.reshape(dataExtended[i],(recordsPerBuffer,-1,samplesPerBlock[i]))
                data[i][buffersCompleted*recordsPerBuffer:(buffersCompleted+1)*recordsPerBuffer] = np.sum(dataBlock, axis=1)

            for i in (np.arange(NtraceA) + NdemodB + NdemodA):
                data[i][buffersCompleted*recordsPerBuffer:(buffersCompleted+1)*recordsPerBuffer] = dataRaw[:recordsPerBuffer,startSample[i]:startSample[i]+samplesPerDemod[i]]

            for i in (np.arange(NtraceB) + NtraceA + NdemodB + NdemodA):
                data[i][buffersCompleted*recordsPerBuffer:(buffersCompleted+1)*recordsPerBuffer] = dataRaw[(channel_number-1)*recordsPerBuffer:channel_number*recordsPerBuffer,startSample[i]:startSample[i]+samplesPerDemod[i]]

            buffersCompleted += 1

            board.postAsyncBuffer(buffer.addr, buffer.size_bytes)

        board.abortAsyncRead()

        if aux_trig:
            board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 1)
            time.sleep(0.1)
            board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 0)


        for i in range(bufferCount):
            buffer = buffers[i]
            buffer.__exit__()

        # Normalize the np.sum and convert data into Volts
        for i in range(NdemodA + NdemodB):
            normalisation = 1 if samplesMissing[i] else 0
            data[i][:,:samplesPerBlock[i]-samplesMissing[i]] /= NumberOfBlocks[i] + normalisation
            data[i][:,samplesPerBlock[i]-samplesMissing[i]:] /= NumberOfBlocks[i]
            data[i] = (data[i] / code - 1) * channelRange
        for i in (np.arange(NtraceA + NtraceB) + NdemodA + NdemodB):
            data[i] = (data[i] / code - 1) * channelRange

        # calculate demodulation tables
        coses=[]
        sines=[]
        numberZeroAdd=[]
        LengthCosine=[]
        for i in range(NdemodA+NdemodB):
            if demodCosinus:
                dem = np.arange(samplesPerBlock[i])
                coses.append(np.cos(2. * math.pi * dem * freq[i] / samplesPerSec))
                sines.append(np.sin(2. * math.pi * dem * freq[i] / samplesPerSec))
            else:
                coses.append(demodFormFile[0])
                sines.append(demodFormFile[1])
                LengthCosine.append(len(coses[i]))

                if len(coses[i]) < samplesPerBlock[i]:
                    numberZeroAdd.append(samplesPerBlock[i]-len(coses[i]))
                    coses[i] = np.concatenate((coses[i],0*np.arange(numberZeroAdd[i])))
                    sines[i] = np.concatenate((sines[i],0*np.arange(numberZeroAdd[i])))
                else:
                    numberZeroAdd.append(0)


        # prepare the structure of the answered array

        if (NdemodA or NdemodB):
            answerTypeDemod = []
            answerTypePower = []
            Nstep = [int((samplesPerDemod[i]/int(samplesPerSec*timestep[i])) if timestep[i] else 1) for i in range(NdemodA+NdemodB)]
            for i in range(NdemodA+NdemodB):
                if i<NdemodA:
                    chanLetter = 'A'
                    zerosDemod = 1 + int(np.floor(np.log10(NdemodA)))
                    index = str(i).zfill(zerosDemod)
                else:
                    chanLetter = 'B'
                    zerosDemod = 1 + int(np.floor(np.log10(NdemodB)))
                    index = str(i-NdemodA).zfill(zerosDemod)
                zerosStep = 1 + int(np.floor(np.log10(Nstep[i])))
                for j in range(Nstep[i]):
                    if Nstep[i]>1:
                        iindex = index + '_' + str(j).zfill(zerosStep)
                    else:
                        iindex = index
                    answerTypeDemod += [(chanLetter + 'I' + iindex, str(data[0].dtype)),
                                        (chanLetter + 'Q' + iindex, str(data[0].dtype))]
                    answerTypePower += [(chanLetter + iindex, str(data[0].dtype))]
        else:
            answerTypeDemod = 'f'
            answerTypePower = 'f'

        if (NtraceA or NtraceB):
            zerosTraceA = 1 + int(np.floor(np.log10(NtraceA))) if NtraceA else 0
            zerosTraceB = 1 + int(np.floor(np.log10(NtraceB))) if NtraceB else 0
            answerTypeTrace = ([('A' + str(i).zfill(zerosTraceA), str(data[0].dtype)) for i in range(NtraceA)]
                              + [('B' + str(i).zfill(zerosTraceB), str(data[0].dtype)) for i in range(NtraceB)] )
            biggerTrace = np.max(samplesPerDemod[NdemodA+NdemodB:])
        else:
            answerTypeTrace = 'f'
            biggerTrace = 0

        if average:
            if Npoints == 0.0:
                answerDemod = np.zeros(1, dtype=answerTypeDemod)
                answerTrace = np.zeros(biggerTrace, dtype=answerTypeTrace)
                answerPower = np.zeros(1, dtype=answerTypePower)

            else:
                answerDemod = np.zeros((1, Npoints), dtype=answerTypeDemod)
                answerTrace = np.zeros((Npoints,biggerTrace), dtype=answerTypeTrace)
                answerPower = np.zeros((1, Npoints), dtype=answerTypePower)
        else:
            answerDemod = np.zeros(recordsPerCapture, dtype=answerTypeDemod)
            answerTrace = np.zeros((recordsPerCapture, biggerTrace), dtype=answerTypeTrace)
            answerPower = np.zeros(recordsPerCapture, dtype=answerTypePower)

        # Demodulate the data, average them if asked and return the result

        for i in np.arange(NdemodA+NdemodB):
            if i<NdemodA:
                chanLetter = 'A'
                zerosDemod = 1 + int(np.floor(np.log10(NdemodA)))
                index = str(i).zfill(zerosDemod)
                powerbool = power[0]
            else:
                chanLetter = 'B'
                zerosDemod = 1 + int(np.floor(np.log10(NdemodB)))
                index = str(i-NdemodA).zfill(zerosDemod)
                powerbool = power[1]
            zerosStep = 1 + int(np.floor(np.log10(Nstep[i])))
            angle = -2 * np.pi * freq[i] * startSample[i] / samplesPerSec
            if (average and Npoints == 0.0):
                if powerbool:
                    ansP = 4 *np.mean(np.mean(((data[i]*coses[i])**2+(data[i]*sines[i])**2).reshape(recordsPerCapture,Nstep[i], -1), axis=0),axis=1)
                data[i] = np.mean(data[i], axis=0)
                ansI = 2 * np.mean((data[i]*coses[i]).reshape(Nstep[i], -1), axis=1)
                ansQ = 2 * np.mean((data[i]*sines[i]).reshape(Nstep[i], -1), axis=1)
                if not demodCosinus:
                    ansI = (LengthCosine[i]+numberZeroAdd[i])/LengthCosine[i]*ansI
                    ansQ = (LengthCosine[i]+numberZeroAdd[i])/LengthCosine[i]*ansQ
                    if powerbool:
                        ansP = (LengthCosine[i]+numberZeroAdd[i])/LengthCosine[i]*ansP
                for j in range(Nstep[i]):
                    if Nstep[i]>1:
                        iindex = index + '_' + str(j).zfill(zerosStep)
                    else:
                        iindex = index
                    answerDemod[chanLetter + 'I' + iindex] = ansI[j] * np.cos(angle) - ansQ[j] * np.sin(angle)
                    answerDemod[chanLetter + 'Q' + iindex] = ansI[j] * np.sin(angle) + ansQ[j] * np.cos(angle)
                    if powerbool:
                        answerPower[chanLetter + iindex] = ansP[j]

            elif average:
                data[i] = data[i].reshape(int(recordsPerCapture/Npoints),Npoints,samplesPerBlock[i])
                if powerbool:
                    ansP = 4 *np.mean(np.mean(((data[i]*coses[i])**2+(data[i]*sines[i])**2).reshape(int(recordsPerCapture/Npoints),Npoints,Nstep[i], -1), axis=0),axis=2)                
                data[i] = np.mean(data[i], axis=0)
                ansI = 2 * np.mean((data[i]*coses[i]).reshape(Npoints,Nstep[i],-1), axis=2)
                ansQ = 2 * np.mean((data[i]*sines[i]).reshape(Npoints,Nstep[i],-1), axis=2)
                if not demodCosinus:
                    ansI = (LengthCosine[i]+numberZeroAdd[i])/LengthCosine[i]*ansI
                    ansQ = (LengthCosine[i]+numberZeroAdd[i])/LengthCosine[i]*ansQ
                    if powerbool:
                        ansP = (LengthCosine[i]+numberZeroAdd[i])/LengthCosine[i]*ansP
                for j in range(Nstep[i]):
                    if Nstep[i]>1:
                        iindex = index + '_' + str(j).zfill(zerosStep)
                    else:
                        iindex = index
                    answerDemod[chanLetter + 'I' + iindex] = ansI[:,j] * np.cos(angle) - ansQ[:,j] * np.sin(angle)
                    answerDemod[chanLetter + 'Q' + iindex] = ansI[:,j] * np.sin(angle) + ansQ[:,j] * np.cos(angle)
                    if powerbool:
                        answerPower[chanLetter + iindex] = ansP[:,j]

            else:
                if powerbool:
                    ansP = 4 * np.mean(((data[i]*coses[i])**2+(data[i]*sines[i])**2).reshape(recordsPerCapture,Nstep[i], -1),axis=2)
                ansI = 2 * np.mean((data[i]*coses[i]).reshape(recordsPerCapture, Nstep[i], -1), axis=2)
                ansQ = 2 * np.mean((data[i]*sines[i]).reshape(recordsPerCapture, Nstep[i], -1), axis=2)
                if not demodCosinus:
                    ansI = (LengthCosine[i]+numberZeroAdd[i])/LengthCosine[i]*ansI
                    ansQ = (LengthCosine[i]+numberZeroAdd[i])/LengthCosine[i]*ansQ
                    if powerbool:
                            ansP = (LengthCosine[i]+numberZeroAdd[i])/LengthCosine[i]*ansP
                for j in range(Nstep[i]):
                    if Nstep[i]>1:
                        iindex = index + '_' + str(j).zfill(zerosStep)
                    else:
                        iindex = index
                    answerDemod[chanLetter + 'I' + iindex] = ansI[:,j] * np.cos(angle) - ansQ[:,j] * np.sin(angle)
                    answerDemod[chanLetter + 'Q' + iindex] = ansI[:,j] * np.sin(angle) + ansQ[:,j] * np.cos(angle)
                    if powerbool:
                        answerPower[chanLetter + iindex] = ansP[:,j]


        for i in (np.arange(NtraceA+NtraceB) + NdemodB+NdemodA):
            if i<NdemodA+NdemodB+NtraceA:
                Tracestring = 'A' + str(i-NdemodA-NdemodB).zfill(zerosTraceA)
            else:
                Tracestring = 'B' + str(i-NdemodA-NdemodB-NtraceA).zfill(zerosTraceB)
            if average:
                if Npoints==0:
                    answerTrace[Tracestring][:samplesPerDemod[i]] = np.mean(data[i], axis=0)
                else:
                    data[i] = data[i].reshape(int(recordsPerCapture/Npoints),Npoints,biggerTrace)
                    answerTrace[Tracestring][:,:samplesPerDemod[i]] = np.mean(data[i], axis=0)
            else:
                answerTrace[Tracestring][:,:samplesPerDemod[i]] = data[i]

        return answerDemod, answerTrace, answerPower


    def get_VNAdemod(self, startaftertrig, duration, recordsPerCapture,
                  recordsPerBuffer, freq, average, Nfreq, NdemodA, NdemodB,
                  demodFormFile,demodCosinus, aux_trig):

        board = ats.Board()

        # Number of samples per record: must be divisible by 32
        samplesPerSec = 500000000.0
        samplesPerTrace = int(samplesPerSec * np.max(np.array(startaftertrig) + np.array(duration)))
        if samplesPerTrace % 32 == 0:
            samplesPerRecord = int(samplesPerTrace)
        else:
            samplesPerRecord = int((samplesPerTrace)/32 + 1)*32

        # Compute the number of bytes per record and per buffer
        channel_number = 2 if ((NdemodA) and (NdemodB)) else 1  # Acquisition on A and B
        memorySize_samples, bitsPerSample = board.getChannelInfo()
        bytesPerSample = (bitsPerSample.value + 7) // 8
        bytesPerRecord = bytesPerSample * samplesPerRecord
        bytesPerBuffer = int(bytesPerRecord * recordsPerBuffer*channel_number)

        # For converting data into volts
        channelRange = 0.4 # Volts
        bitsPerSample = 12
        bitShift = 4
        code = (1 << (bitsPerSample - 1)) - 0.5

        bufferCount = int(round(recordsPerCapture / recordsPerBuffer))
        buffers = []
        for i in range(bufferCount):
            buffers.append(DMABuffer(bytesPerSample, bytesPerBuffer))

        # Set the record size
        board.setRecordSize(0, samplesPerRecord)

        # Configure the number of records in the acquisition
        acquisition_timeout_sec = 10
        board.setRecordCount(int(recordsPerCapture))

        # Calculate the number of buffers in the acquisition
        buffersPerAcquisition = round(recordsPerCapture / recordsPerBuffer)

        channelSelect = 1 if not (NdemodB) else (2 if not (NdemodA) else 3)
        board.beforeAsyncRead(channelSelect,  # Channels A & B
                                  0,
                                  samplesPerRecord,
                                  int(recordsPerBuffer),
                                  int(recordsPerCapture),
                                  ats.ADMA_EXTERNAL_STARTCAPTURE |
                                  ats.ADMA_NPT)

        # Post DMA buffers to board. ATTENTION it is very important not to do "for buffer in buffers"
        for i in range(bufferCount):
            buffer = buffers[i]
            board.postAsyncBuffer(buffer.addr, buffer.size_bytes)

        start = time.perf_counter()  # Keep track of when acquisition started
        board.startCapture()  # Start the acquisition

        if aux_trig:
            time.sleep(0.5)
            board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 1)
            time.sleep(0.1)
            board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 0)

        if time.perf_counter() - start > acquisition_timeout_sec:
            board.abortCapture()
            raise Exception("Error: Capture timeout. Verify trigger")
            time.sleep(10e-3)

        # Preparation of the tables for the demodulation

        startSample = []
        samplesPerDemod = []
        samplesPerBlock = []
        NumberOfBlocks = []
        samplesMissing = []
        data = []
        dataExtended = []

        for i in range(NdemodA*Nfreq+NdemodB*Nfreq):
            startSample.append(int(samplesPerSec * startaftertrig[int(np.floor(i/Nfreq))]) )
            samplesPerDemod.append(int(samplesPerSec * duration[int(np.floor(i/Nfreq))]) )

            if not demodCosinus:
                samplesPerBlock.append(samplesPerDemod[i] )

            else:

                # Check wheter it is possible to cut each record in blocks of size equal
                # to an integer number of periods
                periodsPerBlock = 1

                while (periodsPerBlock * samplesPerSec < freq[i] * samplesPerDemod[i]
                      and periodsPerBlock * samplesPerSec % freq[i]):
                    periodsPerBlock += 1
                samplesPerBlock.append(int(np.minimum(periodsPerBlock * samplesPerSec / freq[i],
                                                         samplesPerDemod[i])) )

            NumberOfBlocks.append(np.divide(samplesPerDemod[i], samplesPerBlock[i]) )
            samplesMissing.append((-samplesPerDemod[i]) % samplesPerBlock[i] )
            # Makes the table that will contain the dataint(np.floor(i/Nfreq))
            data.append(np.empty((recordsPerCapture, samplesPerBlock[i])) )
            dataExtended.append(np.zeros((recordsPerBuffer, samplesPerDemod[i] + samplesMissing[i]),
                                          dtype='uint16') )


        start = time.perf_counter()

        buffersCompleted = 0
        while buffersCompleted < buffersPerAcquisition:

            # Wait for the buffer at the head of the list of available
            # buffers to be filled by the board.
            buffer = buffers[buffersCompleted % len(buffers)]
            board.waitAsyncBufferComplete(buffer.addr, 10000)

            # Process data

            dataRaw = np.reshape(buffer.buffer, (recordsPerBuffer*channel_number, -1))
            dataRaw = dataRaw >> bitShift


            for i in np.arange(NdemodA*Nfreq):
                dataExtended[i][:,:samplesPerDemod[i]] = dataRaw[:recordsPerBuffer,startSample[i]:startSample[i]+samplesPerDemod[i]]
                dataBlock = np.reshape(dataExtended[i],(recordsPerBuffer,-1,samplesPerBlock[i]))
                data[i][buffersCompleted*recordsPerBuffer:(buffersCompleted+1)*recordsPerBuffer] = np.sum(dataBlock, axis=1)

            for i in (np.arange(NdemodB*Nfreq) + NdemodA*Nfreq):
                dataExtended[i][:,:samplesPerDemod[i]] = dataRaw[(channel_number-1)*recordsPerBuffer:channel_number*recordsPerBuffer,startSample[i]:startSample[i]+samplesPerDemod[i]]
                dataBlock = np.reshape(dataExtended[i],(recordsPerBuffer,-1,samplesPerBlock[i]))
                data[i][buffersCompleted*recordsPerBuffer:(buffersCompleted+1)*recordsPerBuffer] = np.sum(dataBlock, axis=1)

            buffersCompleted += 1

            board.postAsyncBuffer(buffer.addr, buffer.size_bytes)

        board.abortAsyncRead()

        if aux_trig:
            #time.sleep(0.5)
            board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 1)
            time.sleep(0.1)
            board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 0)

        for i in range(bufferCount):
            buffer = buffers[i]
            buffer.__exit__()

        # Normalize the np.sum and convert data into Volts
        for i in range(NdemodA*Nfreq + NdemodB*Nfreq):
            normalisation = 1 if samplesMissing[i] else 0
            data[i][:,:samplesPerBlock[i]-samplesMissing[i]] /= NumberOfBlocks[i] + normalisation
            data[i][:,samplesPerBlock[i]-samplesMissing[i]:] /= NumberOfBlocks[i]
            data[i] = (data[i] / code - 1) * channelRange

        # calculate demodulation tables
        coses=[]
        sines=[]
        numberZeroAdd=[]
        LengthCosine=[]
        for i in range(NdemodA*Nfreq+NdemodB*Nfreq):
            if demodCosinus:
                dem = np.arange(samplesPerBlock[i])
                coses.append(np.cos(2. * math.pi * dem * freq[i] / samplesPerSec))
                sines.append(np.sin(2. * math.pi * dem * freq[i] / samplesPerSec))
            else:
                coses.append(demodFormFile[2*i])
                sines.append(demodFormFile[2*i+1])

                if len(coses[i]) < samplesPerBlock[i]:
                    numberZeroAdd.append(samplesPerBlock[i]-len(coses[i]))
                    LengthCosine.append(len(coses[i]))
                    coses[i] = np.concatenate((coses[i],0*np.arange(numberZeroAdd[i])))
                    sines[i] = np.concatenate((sines[i],0*np.arange(numberZeroAdd[i])))


        # prepare the structure of the answered array

        if (NdemodA or NdemodB):
            answerTypeDemod = []
            for i in range(NdemodA+NdemodB):
                if i<NdemodA*Nfreq:
                    chanLetter = 'A'
                    zerosDemod = 1 + int(np.floor(np.log10(NdemodA)))
                    index = str(i).zfill(zerosDemod)
                else:
                    chanLetter = 'B'
                    zerosDemod = 1 + int(np.floor(np.log10(NdemodB)))
                    index = str(i-NdemodA).zfill(zerosDemod)
                answerTypeDemod += [(chanLetter + 'I' + index, str(data[0].dtype)),
                                    (chanLetter + 'Q' + index, str(data[0].dtype))]
        else:
            answerTypeDemod = 'f'


        if average:
            answerDemod = np.zeros((1, Nfreq), dtype=answerTypeDemod)

        else:
            answerDemod = np.zeros((int(recordsPerCapture/Nfreq),Nfreq), dtype=answerTypeDemod)


        # Demodulate the data, average them if asked and return the result

        for i in np.arange(NdemodA+NdemodB):
            if i<NdemodA:
                chanLetter = 'A'
                zerosDemod = 1 + int(np.floor(np.log10(NdemodA)))
                index = str(i).zfill(zerosDemod)
            else:
                chanLetter = 'B'
                zerosDemod = 1 + int(np.floor(np.log10(NdemodB)))
                index = str(i-NdemodA).zfill(zerosDemod)
            angle = -2 * np.pi * freq[i] * startSample[i] / samplesPerSec

            for j in np.arange(Nfreq):
                if average:
                    dataFreq = np.array([data[i*Nfreq+j][k*Nfreq+j,:] for k in np.arange(int(recordsPerCapture/Nfreq))])
                    dataFreq = np.mean(dataFreq, axis=0)
                    ansI = 2 * np.mean((dataFreq*coses[i]), axis=0)
                    ansQ = 2 * np.mean((dataFreq*sines[i]), axis=0)
                    if not demodCosinus:
                        ansI = (LengthCosine[i]+numberZeroAdd[i])/LengthCosine[i]*ansI
                        ansQ = (LengthCosine[i]+numberZeroAdd[i])/LengthCosine[i]*ansQ

                    answerDemod[chanLetter + 'I' + index][0,j] = ansI * np.cos(angle) - ansQ * np.sin(angle)
                    answerDemod[chanLetter + 'Q' + index][0,j] = ansI * np.sin(angle) + ansQ * np.cos(angle)

                else:
                    dataFreq = np.array([data[i*Nfreq+j][k*Nfreq+j,:] for k in np.arange(int(recordsPerCapture/Nfreq))])
                    ansI = 2 * np.mean((dataFreq*coses[i]).reshape(int(recordsPerCapture/Nfreq), -1), axis=1)
                    ansQ = 2 * np.mean((dataFreq*sines[i]).reshape(int(recordsPerCapture/Nfreq), -1), axis=1)
                    if not demodCosinus:
                        ansI = (LengthCosine[i]+numberZeroAdd[i])/LengthCosine[i]*ansI
                        ansQ = (LengthCosine[i]+numberZeroAdd[i])/LengthCosine[i]*ansQ

                    answerDemod[chanLetter + 'I' + index][:,j] = ansI[:] * np.cos(angle) - ansQ[:] * np.sin(angle)
                    answerDemod[chanLetter + 'Q' + index][:,j] = ansI[:] * np.sin(angle) + ansQ[:] * np.cos(angle)


        return answerDemod
    
    
    def get_FFT(self, startaftertrig, duration, recordsPerCapture,
                  recordsPerBuffer, average,
                  NtraceA, NtraceB,Npoints,powerPhase):

        board = ats.Board()

        # Number of samples per record: must be divisible by 32
        samplesPerSec = 500000000.0
        samplesPerTrace = int(samplesPerSec * np.max(np.array(startaftertrig) + np.array(duration)))
        if samplesPerTrace % 32 == 0:
            samplesPerRecord = int(samplesPerTrace)
        else:
            samplesPerRecord = int((samplesPerTrace)/32 + 1)*32

        # Compute the number of bytes per record and per buffer
        channel_number = 2 if (NtraceA and NtraceB) else 1  # Acquisition on A and B
        memorySize_samples, bitsPerSample = board.getChannelInfo()
        bytesPerSample = (bitsPerSample.value + 7) // 8
        bytesPerRecord = bytesPerSample * samplesPerRecord
        bytesPerBuffer = int(bytesPerRecord * recordsPerBuffer*channel_number)

        # For converting data into volts
        channelRange = 0.4 # Volts
        bitsPerSample = 12
        bitShift = 4
        code = (1 << (bitsPerSample - 1)) - 0.5

        bufferCount = int(round(recordsPerCapture / recordsPerBuffer))
        buffers = []
        for i in range(bufferCount):
            buffers.append(DMABuffer(bytesPerSample, bytesPerBuffer))

        # Set the record size
        board.setRecordSize(0, samplesPerRecord)

        # Configure the number of records in the acquisition
        acquisition_timeout_sec = 10
        board.setRecordCount(int(recordsPerCapture))

        # Calculate the number of buffers in the acquisition
        buffersPerAcquisition = round(recordsPerCapture / recordsPerBuffer)

        channelSelect = 1 if not NtraceB else (2 if not NtraceA else 3)
        board.beforeAsyncRead(channelSelect,  # Channels A & B
                                  0,
                                  samplesPerRecord,
                                  int(recordsPerBuffer),
                                  int(recordsPerCapture),
                                  ats.ADMA_EXTERNAL_STARTCAPTURE |
                                  ats.ADMA_NPT)

        # Post DMA buffers to board. ATTENTION it is very important not to do "for buffer in buffers"
        for i in range(bufferCount):
            buffer = buffers[i]
            board.postAsyncBuffer(buffer.addr, buffer.size_bytes)

        start = time.perf_counter()  # Keep track of when acquisition started
        board.startCapture()  # Start the acquisition

        if time.perf_counter() - start > acquisition_timeout_sec:
            board.abortCapture()
            raise Exception("Error: Capture timeout. Verify trigger")
            time.sleep(10e-3)

        # Preparation of the tables for the demodulation

        startSample = []
        samplesPerDemod = []
        data = []

        for i in (np.arange(NtraceA + NtraceB)):
            startSample.append(int(samplesPerSec * startaftertrig[i]) )
            samplesPerDemod.append(int(samplesPerSec * duration[i]) )
            data.append(np.empty((recordsPerCapture, samplesPerDemod[i])) )

        start = time.perf_counter()

        buffersCompleted = 0
        while buffersCompleted < buffersPerAcquisition:

            # Wait for the buffer at the head of the list of available
            # buffers to be filled by the board.
            buffer = buffers[buffersCompleted % len(buffers)]
            board.waitAsyncBufferComplete(buffer.addr, 10000)

            # Process data

            dataRaw = np.reshape(buffer.buffer, (recordsPerBuffer*channel_number, -1))
            dataRaw = dataRaw >> bitShift

            for i in (np.arange(NtraceA)):
                data[i][buffersCompleted*recordsPerBuffer:(buffersCompleted+1)*recordsPerBuffer] = dataRaw[:recordsPerBuffer,startSample[i]:startSample[i]+samplesPerDemod[i]]

            for i in (np.arange(NtraceB) + NtraceA):
                data[i][buffersCompleted*recordsPerBuffer:(buffersCompleted+1)*recordsPerBuffer] = dataRaw[(channel_number-1)*recordsPerBuffer:channel_number*recordsPerBuffer,startSample[i]:startSample[i]+samplesPerDemod[i]]

            buffersCompleted += 1

            board.postAsyncBuffer(buffer.addr, buffer.size_bytes)

        board.abortAsyncRead()

        for i in range(bufferCount):
            buffer = buffers[i]
            buffer.__exit__()

        # Convert data into Volts
        for i in (np.arange(NtraceA + NtraceB)):
            data[i] = (data[i] / code - 1) * channelRange

        # prepare the structure of the answered array
        
        if (NtraceA or NtraceB):
            zerosTraceA = 1 + int(np.floor(np.log10(NtraceA))) if NtraceA else 0
            zerosTraceB = 1 + int(np.floor(np.log10(NtraceB))) if NtraceB else 0
            answerTypeTrace = ([('A' + str(i).zfill(zerosTraceA), str(data[0].dtype)) for i in range(NtraceA)]
                              + [('B' + str(i).zfill(zerosTraceB), str(data[0].dtype)) for i in range(NtraceB)] )
            answerTypeFFT = ([('AI' + str(i).zfill(zerosTraceA), str(data[0].dtype)) for i in range(NtraceA)]+
                              [('AQ' + str(i).zfill(zerosTraceA), str(data[0].dtype)) for i in range(NtraceA)]
                              + [('BI' + str(i).zfill(zerosTraceB), str(data[0].dtype))for i in range(NtraceB)]+
                                 [('BQ' + str(i).zfill(zerosTraceB), str(data[0].dtype))for i in range(NtraceB)] )
            biggerTrace = np.max(samplesPerDemod)
            
        if average:
            if Npoints == 0.0:
                answerFFT = np.zeros(biggerTrace//2+1, dtype=answerTypeFFT)
                answerFFTpower = np.zeros(biggerTrace//2+1, dtype=answerTypeTrace)
                answerFFTphase = np.zeros(biggerTrace//2+1, dtype=answerTypeTrace)

            else:
                answerFFT = np.zeros((Npoints,biggerTrace//2+1), dtype=answerTypeFFT)
                answerFFTpower = np.zeros((Npoints,biggerTrace//2+1), dtype=answerTypeTrace)
                answerFFTphase = np.zeros((Npoints,biggerTrace//2+1), dtype=answerTypeTrace)
        else:
            answerFFT = np.zeros((recordsPerCapture, biggerTrace//2+1), dtype=answerTypeFFT)
            answerFFTpower = np.zeros((recordsPerCapture, biggerTrace//2+1), dtype=answerTypeTrace)
            answerFFTphase = np.zeros((recordsPerCapture, biggerTrace//2+1), dtype=answerTypeTrace)
        answerFreq = np.zeros(biggerTrace//2+1, dtype=answerTypeTrace)
        # FFT of trace, average them if asked and return the result

        for i in (np.arange(NtraceA+NtraceB)):
            if i<NtraceA:
                Tracestring = 'A' + str(i).zfill(zerosTraceA)
                channel = 'A'
                PP_bool = powerPhase[0]
                index = str(i).zfill(zerosTraceA)
            else:
                Tracestring = 'B' + str(i-NtraceA).zfill(zerosTraceB)
                PP_bool = powerPhase[1]
                channel = 'B'
                index = str(i-NtraceA).zfill(zerosTraceB)
            if average:
                if Npoints==0:
                    spectrum =np.fft.rfft(np.mean(data[i], axis=0))
                    answerFFT[channel+'I' + index][:samplesPerDemod[i]//2+1] = np.real(spectrum)
                    answerFFT[channel+'Q' + index][:samplesPerDemod[i]//2+1] = np.imag(spectrum)
                    answerFreq[Tracestring][:samplesPerDemod[i]//2+1] =np.fft.rfftfreq(samplesPerDemod[i],2)
                    if PP_bool:
                        spectrum = np.fft.rfft(data[i])
                        answerFFTpower[Tracestring][:samplesPerDemod[i]//2+1] = np.mean(np.abs(spectrum),axis=0)
                        answerFFTphase[Tracestring][:samplesPerDemod[i]//2+1] = np.mean(np.angle(spectrum),axis=0)
                        
                else:
                    data[i] = data[i].reshape(int(recordsPerCapture/Npoints),Npoints,biggerTrace//2+1)
                    spectrum = np.fft.rfft(np.mean(data[i], axis=0))
                    answerFFT[channel+'I' + index][:,:samplesPerDemod[i]//2+1] =np.real(spectrum,axes=-1)
                    answerFFT[channel+'Q' + index][:,:samplesPerDemod[i]//2+1] =np.imag(spectrum)
                    answerFreq[Tracestring][:samplesPerDemod[i]//2+1] =np.fft.rfftfreq(samplesPerDemod[i],2)
                    if PP_bool:
                        spectrum = np.fft.rfft(data[i])
                        answerFFTpower[Tracestring][:,:samplesPerDemod[i]//2+1] = np.mean(np.abs(spectrum),axis=0)
                        answerFFTphase[Tracestring][:,:samplesPerDemod[i]//2+1] = np.mean(np.angle(spectrum),axis=0)
                     
            else:
                spectrum =np.fft.rfft(data[i])
                answerFFT[channel+'I' + index][:,:samplesPerDemod[i]//2+1] =np.real(spectrum)
                answerFFT[channel+'Q' + index][:,:samplesPerDemod[i]//2+1] =np.imag(spectrum)
                answerFreq[Tracestring][:samplesPerDemod[i]//2+1] =np.fft.rfftfreq(samplesPerDemod[i],2)
                if PP_bool:
                        answerFFTpower[Tracestring][:,samplesPerDemod[i]//2+1] = np.abs(spectrum)
                        answerFFTphase[Tracestring][:,:samplesPerDemod[i]//2+1] = np.angle(spectrum)
                     
        return answerFFT,answerFreq,answerFFTpower,answerFFTphase

    def configure_board_irt(self, sampling_freq, trigger_level, active_channels, record_length,
                                  nof_records_tot, records_per_buf, offset_start):
        board = self.board
        trigger_range = 5
        samples_per_sec = float(sampling_freq) * 1_000_000 # sampling_freq in MS/s

        self.INPUT_RANGE = 400 # mV
        self.CODE_TO_V   = (self.INPUT_RANGE / 1000) / 2**15

        if not self.clock_set:
            board.setCaptureClock(ats.EXTERNAL_CLOCK_10MHz_REF,
                                  500000000,
                                  ats.CLOCK_EDGE_RISING,
                                  0)

            self.clock_set = True

            board.inputControl(ats.CHANNEL_A,
                               ats.DC_COUPLING,
                               ats.INPUT_RANGE_PM_400_MV,
                               ats.IMPEDANCE_50_OHM)

            board.setBWLimit(ats.CHANNEL_A, 0)

            board.inputControl(ats.CHANNEL_B,
                               ats.DC_COUPLING,
                               ats.INPUT_RANGE_PM_400_MV,
                               ats.IMPEDANCE_50_OHM)

            board.setBWLimit(ats.CHANNEL_B, 0)

            trigger_code = int(128 + 127 * trigger_level / trigger_range)
            board.setTriggerOperation(ats.TRIG_ENGINE_OP_J,
                                      ats.TRIG_ENGINE_J,
                                      ats.TRIG_EXTERNAL,
                                      ats.TRIGGER_SLOPE_POSITIVE,
                                      trigger_code,
                                      ats.TRIG_ENGINE_K,
                                      ats.TRIG_DISABLE,
                                      ats.TRIGGER_SLOPE_POSITIVE,
                                      128)

            if trigger_range == 5:
                board.setExternalTrigger(ats.DC_COUPLING,
                                         ats.ETR_5V)
            else:
                board.setExternalTrigger(ats.DC_COUPLING,
                                         ats.ETR_2V5)

            trigger_delay_sec = offset_start / samples_per_sec
            trigger_delay_samples = int(trigger_delay_sec * samples_per_sec + 0.5)
            board.setTriggerDelay(trigger_delay_samples)
            board.setTriggerTimeOut(0)
            board.configureAuxIO(ats.AUX_OUT_TRIGGER, 0)

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
        nof_buffers = int(math.ceil(nof_records / records_per_buf))

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

        buffers = []
        bytes_per_sample = 2
        bytes_per_buffer = bytes_per_sample * max(record_length) * records_per_buf
        for i in range(nof_buffers):
            buffers.append(DMABuffer(bytes_per_sample, bytes_per_buffer))

        nof_active_channels = 2 if ((N_demod_A or N_trace_A) and (N_demod_B or N_trace_B)) else 1

        # Set the record size
        self.board.setRecordSize(0, max(record_length))

        # Configure the number of records in the acquisition
        self.board.setRecordCount(int(records_per_buf * nof_buffers))

        channelSelect = 1 if not N_demod_B else (2 if not N_demod_A else 3)
        self.board.beforeAsyncRead(channelSelect, # Channels A & B
                                   0,
                                   max(record_length),
                                   int(records_per_buf),
                                   int(nof_records),
                                   ats.ADMA_EXTERNAL_STARTCAPTURE |
                                   ats.ADMA_NPT)

        # Post DMA buffers to board. ATTENTION it is very important not to do "for buffer in buffers"
        for i in range(nof_buffers):
            buffer = buffers[i]
            self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)

        self.board.startCapture() # Start the acquisition
        print("Starting data acquisition")

        # 100ms aux trig pulse
        if enable_aux_trig:
            self.board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 1)
            time.sleep(0.1)
            self.board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 0)

        try:
            buffers_completed = 0
            while buffers_completed < nof_buffers:
                # Wait for the buffer at the head of the list of
                # available buffers to be filled by the board.
                buffer = buffers[buffers_completed % len(buffers)]
                self.board.waitAsyncBufferComplete(buffer.addr, timeout)

                # Process data
                buffer_data = np.reshape(buffer.buffer, (records_per_buf*nof_active_channels, -1))

                for i, ch in enumerate(active_channels):
                    data = buffer_data[i*records_per_buf:(i+1)*records_per_buf,:record_length[ch]]

                    start = records_per_buf *  buffers_completed
                    stop  = records_per_buf * (buffers_completed+1)
                    stop2 = records_per_buf

                    # Handle last buffer
                    if nof_records // records_per_buf == buffers_completed:
                        stop2 = nof_records % records_per_buf
                    
                    records_per_experiment = stop2 // nof_experiments

                    # Save raw data to process later
                    if defer_process:
                        data_tot[ch][buffers_completed]     = data
                        buf_params[ch][buffers_completed,:] = np.array([start, stop, stop2, records_per_experiment])
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
                
                buffers_completed += 1

                self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)

        except Exception as e:
            self.board.abortAsyncRead()
            raise e

        # Stop the data acquisition
        print("Stopping data acquisition")
        self.board.abortAsyncRead()

        # 100ms aux trig pulse
        if enable_aux_trig:
            self.board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 1)
            time.sleep(0.1)
            self.board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 0)

        for i in range(nof_buffers):
            buffer = buffers[i]
            buffer.__exit__()

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
