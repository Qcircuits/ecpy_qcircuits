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
import math
import time
import logging

logger = logging.getLogger(__name__)

from ..dll_tools import DllInstrument

try:
    from . import atsapi as ats
except FileNotFoundError:
    logger.info("Couldn't find the Alazar DLL, please install the driver "
                "if you want to use it.")



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

        start = time.clock()  # Keep track of when acquisition started
        board.startCapture()  # Start the acquisition

        if aux_trig:
            time.sleep(1)
            board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 1)
            time.sleep(0.1)
            board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 0)

        if time.clock() - start > acquisition_timeout_sec:
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

        start = time.clock()

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

        start = time.clock()  # Keep track of when acquisition started
        board.startCapture()  # Start the acquisition

        if aux_trig:
            time.sleep(0.5)
            board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 1)
            time.sleep(0.1)
            board.configureAuxIO(ats.AUX_OUT_SERIAL_DATA, 0)

        if time.clock() - start > acquisition_timeout_sec:
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


        start = time.clock()

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

        start = time.clock()  # Keep track of when acquisition started
        board.startCapture()  # Start the acquisition

        if time.clock() - start > acquisition_timeout_sec:
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

        start = time.clock()

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