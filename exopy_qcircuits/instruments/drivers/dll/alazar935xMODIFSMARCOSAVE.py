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

import glob
import re

from ..dll_tools import DllInstrument
from . import atsapi as ats


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
        board.setCaptureClock(ats.EXTERNAL_CLOCK_10MHz_REF,
                                  500000000,
                                  ats.CLOCK_EDGE_RISING,
                                  0)
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
                  NtraceA, NtraceB, Npoints):
        tempsDemodTotalDebut=time.clock()
        
        board = ats.Board()

        # Number of samples per record: must be divisible by 32
        samplesPerSec = 500000000.0
        tt1=time.clock()
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

        if time.clock() - start > acquisition_timeout_sec:
            board.abortCapture()
            raise Exception("Error: Capture timeout. Verify trigger")
            time.sleep(10e-3)
        
        # Preparation of the tables for the demodulation
        
        startSample = []
        samplesPerDemod = []
        data = []
        
        for i in range(NdemodA + NdemodB):
            startSample.append(int(samplesPerSec * startaftertrig[i]) )
            samplesPerDemod.append(int(samplesPerSec * duration[i]) )
            data.append(np.empty((recordsPerCapture, samplesPerDemod[i])) )

        start = time.clock()
        
        RecupBufferDebut=time.clock() 
        LecturePureBufferDuree=0
        buffersCompleted = 0
        tt2=time.clock()
        print("!!!!!!!!!", tt2-tt1)
        while buffersCompleted < buffersPerAcquisition:

            # Wait for the buffer at the head of the list of available
            # buffers to be filled by the board.
            bufferReadBegin=time.clock()
            buffer = buffers[buffersCompleted % len(buffers)]
            board.waitAsyncBufferComplete(buffer.addr, 10000)
            bufferReadEnd=time.clock()
            LecturePureBufferDuree=LecturePureBufferDuree+bufferReadEnd-bufferReadBegin
            # Process data

            dataRaw = np.reshape(buffer.buffer, (recordsPerBuffer*channel_number, -1))
            dataRaw = dataRaw >> bitShift

            for i in np.arange(NdemodA):
                #dataExtended[i][:,:samplesPerDemod[i]] = dataRaw[:recordsPerBuffer,startSample[i]:startSample[i]+samplesPerDemod[i]]
                #dataBlock = np.reshape(dataExtended[i],(recordsPerBuffer,-1,samplesPerBlock[i]))
                #data[i][buffersCompleted*recordsPerBuffer:(buffersCompleted+1)*recordsPerBuffer] = np.sum(dataBlock, axis=1)
                # On prends la trace pure ---------------
                data[i][buffersCompleted*recordsPerBuffer:(buffersCompleted+1)*recordsPerBuffer] = dataRaw[:recordsPerBuffer,startSample[i]:startSample[i]+samplesPerDemod[i]]

            for i in (np.arange(NdemodB) + NdemodA):
                #dataExtended[i][:,:samplesPerDemod[i]] = dataRaw[(channel_number-1)*recordsPerBuffer:channel_number*recordsPerBuffer,startSample[i]:startSample[i]+samplesPerDemod[i]]
                #dataBlock = np.reshape(dataExtended[i],(recordsPerBuffer,-1,samplesPerBlock[i]))
                #data[i][buffersCompleted*recordsPerBuffer:(buffersCompleted+1)*recordsPerBuffer] = np.sum(dataBlock, axis=1)
                data[i][buffersCompleted*recordsPerBuffer:(buffersCompleted+1)*recordsPerBuffer] = dataRaw[(channel_number-1)*recordsPerBuffer:channel_number*recordsPerBuffer,startSample[i]:startSample[i]+samplesPerDemod[i]]
            
            buffersCompleted += 1

            board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
        LecturePureBufferDuree=(LecturePureBufferDuree)*pow(10,3)    
        RecupBufferFin=time.clock()
        
        board.abortAsyncRead()

        for i in range(bufferCount):
            buffer = buffers[i]
            buffer.__exit__()
            
        
        # On normalise tout. Maintenant comme on va faire la méthode du produit
        # matriciel on a plus de distinction trace/demod ici.
        
        tempsCalculDebut=time.clock()
        
        tempsMiseEnFormeDebut=time.clock()
        for i in (np.arange(NdemodA + NdemodB)):
            data[i] = (data[i] / code - 1) * channelRange
        tempsMiseEnFormeFin=time.clock()
        
        print("Temps mise en forme : ", tempsMiseEnFormeFin-tempsMiseEnFormeDebut)
        
        tempsCreationCosDebut=time.clock()
        B=[]
        coses=[]
        sines=[]
        for i in range(NdemodA+NdemodB):
            #dem = np.arange(samplesPerBlock[i])
            dem = np.arange(samplesPerDemod[i])
            coses.append(np.cos(2. * math.pi * dem * freq[i] / samplesPerSec))
            sines.append(np.sin(2. * math.pi * dem * freq[i] / samplesPerSec))
            # Juste pour check que ça fait le taf :
            #coses.append(dem)
            #sines.append(dem*10000)
            B.append(np.zeros((recordsPerCapture,2)))
            
            coses[i]=np.asarray(coses[i])
            sines[i]=np.asarray(sines[i])
            B[i]=np.transpose(np.asarray([coses[i], sines[i]]))
        tempsCreationCosFin=time.clock()
        print("Temps Creation et mise en forme cos", tempsCreationCosFin-tempsCreationCosDebut)
        
        C=[]
        
        tempsCalculProdDebut=time.clock()
        for i in range(NdemodA+NdemodB):
            C.append(np.empty((recordsPerCapture,2)))
            C[i]=np.dot(data[i],B[i]) # Calcul C=A*B pour obtenir les quadratures I et Q
        tempsCalculProdFin=time.clock()
        
        print("Temps calcul produit duree", tempsCalculProdFin-tempsCalculProdDebut)
        tempsCalculFin=time.clock()
        # A ce stade on a les quadratures !

        # On simule une écriture des quadratures sur un fichier :
        tempsEcritureDebut=time.clock()
        name_fichier="D:\\data\\Marco\\resultat.txt"
        mon_fichier = open(name_fichier, "w")
        
        for i in range(NdemodA+NdemodB):
            c_shape=C[i].shape
            for k1 in range(c_shape[0]):
                for k2 in range(c_shape[1]):
                    mon_fichier.write(str(C[i][k1,k2])+ " ")
                mon_fichier.write("\n")
        tempsEcritureFin=time.clock()
        
        answerDemod=0
        answerTrace=0
        
        tempsDemodTotalFin=time.clock()
        
        tempsDemodTotalDuree=(tempsDemodTotalFin-tempsDemodTotalDebut)*pow(10,3)
        tempsEcritureDuree=(tempsEcritureFin-tempsEcritureDebut)*pow(10,3)
        RecupBufferDuree=(RecupBufferFin-RecupBufferDebut)*pow(10,3)
        tempsCalculDuree=(tempsCalculFin-tempsCalculDebut)*pow(10,3)
        
        print("Temps lecture des buffers (pure)", LecturePureBufferDuree)
        print("Temps lecture buffer + mise en forme data", RecupBufferDuree)
        print("Temps calcul : ",tempsCalculDuree)
        print("Temps écriture : ",tempsEcritureDuree)
        print("Temps total démodulation (écriture incluse) : ",tempsDemodTotalDuree)
        mon_fichier.close()        
        
        
        
        ''' On écrit tous sur un fichier : '''
        
        nbDuration=10
        nbIQ=20
        name_fichier="D:\\data\\Marco\\perfs\\FctDuration.txt"
        
        listFile=glob.glob(name_fichier)
        # On récupère tous les fichiers ayant ce nom. Si il n'y en a pas on met on le crée avec l'entête
        
        if(len(listFile)==0):
            mon_fichier = open(name_fichier, "w")
            mon_fichier.write("Repetition [us]\t Duration\t Nb Traces\t Nb IQ\t Cible\t Demod tot\t Pur Buffer \t Tot Buffer \t Calcul\t Ecriture\n")
        else:
            mon_fichier = open(name_fichier, "a")
            mon_fichier.write("\n")
        
        repetition=10 # repetition en µs
        temps_cible=repetition*recordsPerCapture*pow(10,-3)
        mon_fichier.write(str(repetition)+ "\t" + str(duration[0]*pow(10,3)) + "\t" + str(recordsPerCapture) +"\t" + str(Npoints) + "\t" + str(temps_cible) + "\t" + str(tempsDemodTotalDuree) + "\t" + str(RecupBufferDuree) + "\t" + str(tempsCalculDuree) + "\t" + str(tempsEcritureDuree) +"\n") 
        
        
        mon_fichier.close()
        
        return answerDemod, answerTrace