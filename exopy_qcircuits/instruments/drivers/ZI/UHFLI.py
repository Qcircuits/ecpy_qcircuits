# -*- coding: utf-8 -*-
"""This module defines drivers for UHFLI using Zhinst Library.

:Contains:
    UHFLI

Python package zhinst from Zurick Instruments need to be install 

"""

from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
from ..driver_tools import (InstrIOError, instrument_property,
                            secure_communication)
import sys
from subprocess import call
import ctypes
import os
from inspect import cleandoc
import math

import numpy as np
import time

from ..ZI_tools import ZIInstrument


import zhinst.utils

class UHFLI(ZIInstrument):
    
    def __init__(self,connection_info, caching_allowed=True,
                 caching_permissions={}):
        super(UHFLI, self).__init__(connection_info, caching_allowed,
                                             caching_permissions)
        self.awgModule = None
        self.required_devtype='.*LI'
        self.required_options=['AWG','DIG']
        
    def close_connection(self):
        
        if self.awgModule:
            if self.daq:
                self.daq.setInt('/%s/awgs/0/enable' %self.device, 0)
                self.awgModule.finish()
                self.awgModule.clear()
            
    def set_general_setting(self):
        general_setting = [['/%s/demods/*/enable' % self.device, 0],
                       ['/%s/scopes/*/enable' % self.device, 0]]
        self.daq.set(general_setting)
        self.daq.sync()
        
    def TransfertSequence(self,awg_program):
        # Transfer the AWG sequence program. Compilation starts automatically.
        self.awgModule.set('awgModule/compiler/sourcestring', awg_program)
        
        while self.awgModule.getInt('awgModule/compiler/status') == -1:
            time.sleep(0.1)

        if self.awgModule.getInt('awgModule/compiler/status') == 1:
        # compilation failed, raise an exception
            raise Exception(self.awgModule.getString('awgModule/compiler/statusstring'))
        else:
            if self.awgModule.getInt('awgModule/compiler/status') == 2:
                print("Compilation successful with warnings, will upload the program to the instrument.")
                print("Compiler warning: ",
                      self.awgModule.getString('awgModule/compiler/statusstring'))
        # wait for waveform upload to finish
        i = 0
        while self.awgModule.getDouble('awgModule/progress') < 1.0:
            time.sleep(0.1)
            i += 1
        
    
    def get_scope_demod(self,samplingRate, duration, delay, recordsPerCapture,
                  freq, average,demod,trace, Npoints,customDemod,demodCosinus,scopeModule):
        
        if 0 in duration[0]:
            channel = [1]
        elif 0 in duration[1]:
            channel=[0]
        else : 
            channel = [0,1]
        #erase the memory of the scope
        scopeModule.set('scopeModule/clearhistory', 1)

        # Tell the module to be ready to acquire data; reset the module's progress to 0.0.
        scopeModule.execute()

        # Enable the scope: Now the scope is ready to record data upon receiving triggers.
        
        self.daq.setInt('/%s/scopes/0/single' % self.device, 0)
        self.daq.setInt('/%s/scopes/0/enable' % self.device, 1)
        self.daq.sync()
    
        start = time.time()
        timeout = recordsPerCapture/100  #[s]
        records = 0        
        dataRecorded = []
        
        
        # Wait until the Scope Module has received and processed the desired number of records.
        while (records < recordsPerCapture):
            time.sleep(0.01)
            records = scopeModule.getInt("scopeModule/records")
            
            # Advanced use: It's possible to read-out data before all records have been recorded (or even before all
            # segments in a multi-segment record have been recorded). Note that complete records are removed from the Scope
            # Module and can not be read out again; the read-out data must be managed by the client code. If a multi-segment
            # record is read-out before all segments have been recorded, the wave data has the same size as the complete
            # data and scope data points currently unacquired segments are equal to 0.
            
            data=scopeModule.read(True)
            if '/%s/scopes/0/wave'%self.device in data.keys():
                dataRecorded.extend(data['/%s/scopes/0/wave' % self.device])
            
            
            if (time.time() - start) > timeout:
                # Break out of the loop if for some reason we're no longer receiving scope data from the device.
                print("\nScope Module did not return {} records after {} s - forcing stop.".format(1, timeout))
                break
        self.daq.setInt('/%s/scopes/0/enable' % self.device, 0)
        
        # Read out the scope data from the module.
        data = scopeModule.read(True)
        if '/%s/scopes/0/wave' %self.device in data.keys():
                dataRecorded.extend(data['/%s/scopes/0/wave' %self.device])
        # Stop the module
        scopeModule.set('scopeModule/clearhistory', 1)
        scopeModule.finish()
        scopeModule.clear()
        #check that no problems occur
        dataUnusable=[]
        num_records = len(dataRecorded)
        for i in range(len(dataRecorded)):
            if dataRecorded[i][0]['flags'] & 1:
                dataUnusable.append(i) 
            if dataRecorded[i][0]['flags'] & 2:
                dataUnusable.append(i) 
            if dataRecorded[i][0]['flags'] & 3:
                dataUnusable.append(i) 

    
        # number max of period in the trace at the frequency freq
        nb =np.array([np.int_(duration[i]*freq[i]) for i in range(2)])
        #number of point we keep for the demodulation
        nbSample= [np.int_(nb[i]*1/freq[i]*samplingRate) for i in range(2)]
        delaySample= np.int_(np.array(delay)*samplingRate)
        
        length = [np.int_(i) for i in np.array(duration)*samplingRate]
        #keep only data with no problem
        tracedata = [[],[]]
        for i in range(num_records):
            if not i in dataUnusable:
                for c in range(len(channel)):
                    tracedata[channel[c]].append(dataRecorded[i][0]['wave'][c][:delaySample[channel[c]]+np.sum(length[channel[c]])])
        if max(len(tracedata[0]),len(tracedata[1]))>=recordsPerCapture:
            for c in channel:
                tracedata[c] = np.array(tracedata[c])[:recordsPerCapture,delaySample[c]:delaySample[c]+np.sum(length[c])]
        else:
            raise Exception("Error: To many data not workable")
            
        datatype = str(dataRecorded[0][0]['wave'].dtype)
                       
        del(dataRecorded)
        
        #demodulation function
        coses=[]
        sines=[]

        for c in range(2):
            if demodCosinus:         
                coses.append(np.cos(np.arange(np.sum(length[c]))*2*np.pi*freq[c]/samplingRate))
                sines.append(np.sin(np.arange(np.sum(length[c]))*2*np.pi*freq[c]/samplingRate))
            else:
                coses.append(customDemod[0])
                sines.append(customDemod[1])
            
        
        if demod:
            answerTypeDemod = []
            for c in channel:
                for i in range(len(duration[c])):
                    answerTypeDemod =answerTypeDemod+ [(str(c+1)+'I_'+str(i),datatype),
                                                       (str(c+1)+'Q_'+str(i),datatype)]
        else: 
            answerTypeDemod= 'f'
        if trace:
            answerTypeTrace=[]
            for c in channel:
                for i in range(len(duration[c])):
                    answerTypeTrace = answerTypeTrace+ [(str(c+1)+'_'+str(i),datatype)]
        else:
            answerTypeTrace = 'f'
                
        
        if average:
            if Npoints == 1 or Npoints == 0:
                answerDemod = np.zeros(1, dtype=answerTypeDemod)
                answerTrace = np.zeros(np.max([np.max(length[0]),np.max(length[1])]), dtype=answerTypeTrace)
            else:
                answerDemod = np.zeros((1, Npoints), dtype=answerTypeDemod)
                answerTrace = np.zeros((Npoints,np.max([np.max(length[0]),np.max(length[1])])), dtype=answerTypeTrace)

        else:
            answerDemod = np.zeros(recordsPerCapture, dtype=answerTypeDemod)
            answerTrace = np.zeros((recordsPerCapture, np.max([np.max(length[0]),np.max(length[1])])), dtype=answerTypeTrace)


        for c in channel:
            if demod[c]:
                start =0
                for i in range(len(duration[c])):
                
                    ansI= 2*np.mean(tracedata[c][:,start:start+nbSample[c][i]]*coses[c][start:start+nbSample[c][i]],axis=1)
                    ansQ= 2*np.mean(tracedata[c][:,start:start+nbSample[c][i]]*sines[c][start:start+nbSample[c][i]],axis=1)
            
                    if Npoints!=1 and Npoints!=0 and average:
                        ansI = ansI.reshape((int(recordsPerCapture/Npoints),Npoints))
                        ansQ = ansQ.reshape((int(recordsPerCapture/Npoints),Npoints))
                        
                        ansI = ansI.mean(axis=0)
                        ansQ = ansQ.mean(axis=0)    
                              
                        answerDemod[str(c+1)+'I_'+str(i)]= ansI
                        answerDemod[str(c+1)+'Q_'+str(i)]= ansQ
                                        
                    elif average and (Npoints==1 or Npoints ==0):
                        ansI = np.mean(ansI,axis=0)
                        ansQ = np.mean(ansQ,axis=0)
                        
                        answerDemod[str(c+1)+'I_'+str(i)]= ansI
                        answerDemod[str(c+1)+'Q_'+str(i)]= ansQ
       
                    else:
                        
                        answerDemod[str(c+1)+'I_'+str(i)]= ansI
                        answerDemod[str(c+1)+'Q_'+str(i)]= ansQ
                    
                    start = start+length[c][i]
                            
        for c in channel:
            if trace[c]:
                start =0
                for i in range(len(duration[c])):
                    trace = (tracedata[c][:,start:start+length[c][i]])
                    if Npoints!=1 and Npoints!=0 and average:
                       
                        trace =(trace.reshape((-1,Npoints,length[c][i]))).mean(axis=0)           
                        answerTrace[str(c+1)+'_'+str(i)][:,:length[c][i]]= trace
                                
                    elif average and (Npoints==1 or Npoints ==0):
                        trace = np.mean(trace,axis=0)
               
                        answerTrace[str(c+1)+'_'+str(i)][:length[c][i]]= trace
                    else:
                        answerTrace[str(c+1)+'_'+str(i)][:,:length[c][i]]= trace
                    start = start+length[c][i]
                    
       
        return answerDemod, answerTrace
        
    @secure_communication()    
    def get_demodLI(self,recordsPerCapture,average,Npoints,channel,demod,powerBool,AWGcontrol):
            
        if ['1'] == channel:
            self.daq.setInt('/%s/demods/%d/enable' % (self.device,demod[0]), 1) # enable the stream data of input1
            self.daq.sync()
        elif ['2'] == channel:
            self.daq.setInt('/%s/demods/%d/enable' % (self.device,demod[1]), 1) # enable the stream data of input2
            self.daq.sync()
        else :
            # enable the stream data of the demodulators 3 and 4
            self.daq.set([['/%s/demods/%d/enable' % (self.device,demod[0]), 1],['/%s/demods/%d/enable' % (self.device,demod[1]), 1]])  
            self.daq.sync()
            time.sleep(0.1)

        if AWGcontrol:
            self.daq.setInt('/%s/awgs/0/enable' %self.device, 1)

        data1x=[];
        data1y=[];
        data2x=[];
        data2y=[];
        time1=[]
        time2=[]
        path1 = '/%s/demods/%d/sample' % (self.device,demod[0])
        path2 = '/%s/demods/%d/sample' % (self.device,demod[1])
        data=self.daq.poll(0.1,500,0x0004,True)
        if '1' in channel:
            if path1 in data.keys():
                data1x= data[path1]['x']
                data1y = data[path1]['y']
                time1 = data[path1]['timestamp']
        if '2' in channel:
            if path2 in data.keys():
                data2x= data[path2]['x']
                data2y = data[path2 ]['y']
                time2 = data[path2]['timestamp']
     #   if math.isnan(np.mean(data1x)) or math.isnan(np.mean(data1y)):
   #             print(str(data))
        while(len(data1x)<recordsPerCapture*('1' in channel) or len(data2x)<recordsPerCapture*('2' in channel)):
            data=self.daq.poll(0.1,500,0x0004,True)
            if '1' in channel:
                if path1 in data.keys():
                    data1x = np.concatenate((data1x,data[path1]['x']))
                    data1y = np.concatenate((data1y,data[path1]['y']))
                    time1 =  np.concatenate((time1,data[path1]['timestamp']))
            if '2' in channel: 
                if path2 in data.keys():
                    data2x = np.concatenate((data2x,data[path2]['x']))
                    data2y = np.concatenate((data2y,data[path2]['y']))
                    time2 =  np.concatenate((time2,data[path2]['timestamp']))
        self.daq.setInt('/%s/demods/%s/enable'% (self.device,demod[0]), 0); # close the stream data of input1
        self.daq.setInt('/%s/demods/%s/enable'% (self.device,demod[1]), 0); # close the stream data of input2
        if AWGcontrol:
            self.daq.setInt('/%s/awgs/0/enable' %self.device, 0)
        if ['1','2'] == channel:
            n=0;
            for i in range(min(len(time1),len(time2))):
                if time1[i]!=time2[i]:
                    n+=1;
            print(str(n) + ' errors in demodulation')
            print(str(len(time1)-len(time2)))
        if '1' in channel:
            if math.isnan(np.mean(data1x)):
                listNan = [math.isnan(i) for i in data1x]
                data1x=np.delete(data1x,np.where(listNan))
                data1y=np.delete(data1y,np.where(listNan))
                print('Warning NaN value detect in data1 :%d NaN value, %d value correct' %(np.sum(listNan),len(data1x)))
                if len(data1x)<recordsPerCapture:
                    recordsPerCapture = len(data1x)
        if '2' in channel:
            if math.isnan(np.mean(data2x)):
                listNan = [math.isnan(i) for i in data2x]
                data2x=np.delete(data2x,np.where(listNan))
                data2y=np.delete(data2y,np.where(listNan))
                print('Warning NaN value detect in data2 :%d NaN value, %d value correct' %(np.sum(listNan),len(data2x)))
                if len(data2x)<recordsPerCapture:
                    recordsPerCapture = len(data2x)
                
        if '1' in channel:
                data1x= data1x[:recordsPerCapture]
                data1y= data1y[:recordsPerCapture]
        if '2' in channel:
                data2x= data2x[:recordsPerCapture]
                data2y= data2y[:recordsPerCapture]
        answerTypeDemod=[];
        if '1' in channel:
            answerTypeDemod = answerTypeDemod+ [('1I',str(data1x.dtype)),
                               ('1Q',str(data1y.dtype))]
            if powerBool :
                answerTypeDemod = answerTypeDemod+ [('1P',str(data1x.dtype))]
        if '2' in channel:
            answerTypeDemod = answerTypeDemod+ [('2I',str(data2x.dtype)),
                               ('2Q',str(data2y.dtype))]
            if powerBool :
                answerTypeDemod = answerTypeDemod+ [('2P',str(data2x.dtype))]
                            
        if average:
            if Npoints==0 or Npoints ==1:
                answerDemod = np.zeros(1, dtype=answerTypeDemod)
            else :
                answerDemod = np.zeros((1, Npoints), dtype=answerTypeDemod)

        else:
            answerDemod = np.zeros(recordsPerCapture, dtype=answerTypeDemod)
                
        if average:
            if Npoints==0 or Npoints ==1:
                if '1' in channel:
                    ans1I = np.mean(data1x)
                    ans1Q = np.mean(data1y)
                    answerDemod['1I']= ans1I
                    answerDemod['1Q']= ans1Q
                    if powerBool:
                        ansR = np.mean(data1x**2+data1y**2)
                        answerDemod['1P']= ansR

                if '2' in channel:
                    ans2I = np.mean(data2x)
                    ans2Q = np.mean(data2y)
                    answerDemod['2I']= ans2I
                    answerDemod['2Q']= ans2Q
                    if powerBool:
                        ansR = np.mean(data2x**2+data2y**2)
                        answerDemod['2P']= ansR
            else :
                if '1' in channel:
                    data1x = data1x.reshape((-1,Npoints))
                    data1y = data1y.reshape((-1,Npoints))                                
                    ans1I = data1x.mean(axis=0)
                    ans1Q = data1y.mean(axis=0)
                    answerDemod['1I']= ans1I
                    answerDemod['1Q']= ans1Q
                    if powerBool:
                        ansR = np.mean((data2x**2+data2y**2).reshape((-1,Npoints)),axis=0)
                        answerDemod['1P']= ansR
                        
                if '2' in channel:
                    data2x = data2x.reshape((-1,Npoints))
                    data2y = data2y.reshape((-1,Npoints))
                    ans2I = data2x.mean(axis=0)
                    ans2Q = data2y.mean(axis=0)
                    answerDemod['2I']= ans2I
                    answerDemod['2Q']= ans2Q
                    if powerBool:
                        ansR = np.mean((data2x**2+data2y**2).reshape((-1,Npoints)),axis=0)
                        answerDemod['2P']= ansR
                
        else:
            if '1' in channel:
                answerDemod['1I']= data1x[:recordsPerCapture]
                answerDemod['1Q']= data1y[:recordsPerCapture]
                if powerBool:
                    ansR = data2x**2+data2y**2
                    answerDemod['1P']= ansR
            if '2' in channel:
                answerDemod['2I']= data2x[:recordsPerCapture]
                answerDemod['2Q']= data2y[:recordsPerCapture]
                if powerBool:
                    ansR = data2x**2+data2y**2
                    answerDemod['2P']= ansR
        return answerDemod


    def get_DAQmodule(self, DAM, dimensions, signalID,signal_paths):
        data={i:[] for i in signalID}
        # Start recording data.
        DAM.set('dataAcquisitionModule/endless', 0);
        t0 = time.time()
        # Record data in a loop with timeout.
        timeout =dimensions[0]*dimensions[1]*dimensions[2]*dimensions[3]*0.001+10
        DAM.execute()
        #while not DAM.finished():
        while not DAM.finished():
            if time.time() - t0 > timeout:
                raise Exception("Timeout after {} s - recording not complete.".format(timeout))
            
            data_read = DAM.read(True)
            for sp,sid in np.transpose([signal_paths,signalID]):
                if sp in data_read.keys():
                    for d in data_read[sp]:
                        if d['header']['flags'] & 1:
                            data[sid].append(d['value'])
        DAM.finish()
        # There may be new data between the last read() and calling finished().
        data_read = DAM.read(True)
        for sp,sid in np.transpose([signal_paths,signalID]):
                if sp in data_read.keys():
                    for d in data_read[sp]:
                        if d['header']['flags'] & 1:
                            data[sid].append(d['value'])
        DAM.clear()
    
        answerTypeGrid=[]
        for sid  in signalID:
            answerTypeGrid = answerTypeGrid+ [(sid,str(data[sid][0][0][0].dtype))]
              

        answerDAM = np.zeros((dimensions[0],dimensions[1],dimensions[2]), dtype=answerTypeGrid)
        
        for sid in signalID:
            answerDAM[sid] = data[sid]
        return answerDAM

