
## Functions to calculate metrics and write them down

from math import *

from MITgcmutils import rdmds

from netCDF4 import Dataset

import numpy as np

import os 

import pylab as pl

import scipy.io

import scipy as spy

import sys

lib_path = os.path.abspath('/ocean/kramosmu/Building_canyon/BuildCanyon/PythonModulesMITgcm') # Add absolute path to my python scripts
sys.path.append(lib_path)

import ReadOutTools_MITgcm as rout 


#---------------------------------------------------------------------------------------------------------------------------
def getDatasets(expPath, runName):
    '''Specify the experiment and run from which to analyse state and ptracers output.
    expName : (string) Path to experiment folder. E.g. '/ocean/kramosmu/MITgcm/TracerExperiments/BARKLEY', etc.
    runName : (string) Folder name of the run. E.g. 'run01', 'run10', etc
    '''
    Grid =   "%s/%s/gridGlob.nc" %(expPath,runName)
    GridOut = Dataset(Grid)

    State =  "%s/%s/stateGlob.nc" %(expPath,runName)
    StateOut = Dataset(State)

    Ptracers =  "%s/%s/ptracersGlob.nc" %(expPath,runName)
    PtracersOut = Dataset(Ptracers)
    
    return (Grid, GridOut, State,StateOut,Ptracers, PtracersOut)


#---------------------------------------------------------------------------------------------------------------------------


def getProfile(Tr,yi,xi,nz0=0,nzf=90):
    '''Slice tracer profile at x,y = xi,yi form depth index k=nz0 to k=nzf. Default values are nz0=0 (surface)
    and nzf = 89, bottom. Tr is a time slice (3D) of the tracer field'''
    IniProf = Tr[nz0:nzf,yi,xi]
    return IniProf


#---------------------------------------------------------------------------------------------------------------------------


def maskExpand(mask,Tr):
    
    '''Expand the dimensions of mask to fit those of Tr. mask should have one dimension less than Tr (time axis). 
    It adds a dimension before the first one.'''
    
    mask_expand = np.expand_dims(mask,0)
    
    mask_expand = mask_expand + np.zeros(Tr.shape)
    
    return mask_expand

    
#---------------------------------------------------------------------------------------------------------------------------


def howMuchWaterX(Tr,MaskC,nzlim,rA,hFacC,drF,yin,zfin,xi,yi):
    '''
    INPUT----------------------------------------------------------------------------------------------------------------
    Tr    : Array with concentration values for a tracer. Until this function is more general, this should be size 19x90x360x360
    MaskC : Land mask for tracer
    nzlim : The nz index under which to look for water properties
    rA    : Area of cell faces at C points (360x360)
    fFacC : Fraction of open cell (90x360x360)
    drF   : Distance between cell faces (90)
    yin   : across-shore index of shelf break
    zfin  : shelf break index + 1 
    xi    : initial profile x index
    yi    : initial profile y index
    
    OUTPUT----------------------------------------------------------------------------------------------------------------
    VolWaterHighConc =  Array with the volume of water over the shelf [:,:30,227:,:] at every time output.
    Total_Tracer =  Array with the mass of tracer (m^3*[C]*l/m^3) at each x-position over the shelf [:,:30,227:,:] at 
                    every time output. Total mass of tracer at xx on the shelf.
                                                
    -----------------------------------------------------------------------------------------------------------------------
    '''
    maskExp = maskExpand(MaskC,Tr)

    TrMask=np.ma.array(Tr,mask=maskExp)   
    
    trlim = TrMask[0,nzlim,yi,xi]
    
    print('tracer limit concentration is: ',trlim)
    
    WaterX = 0
    
    # mask cells with tracer concentration < trlim on shelf
    HighConc_Masked = np.ma.masked_less(TrMask[:,:zfin,yin:,:], trlim) 
    HighConc_Mask = HighConc_Masked.mask
    
    #Get volume of water of cells with relatively high concentration
    rA_exp = np.expand_dims(rA[yin:,:],0)
    drF_exp = np.expand_dims(np.expand_dims(drF[:zfin],1),1)
    rA_exp = rA_exp + np.zeros(hFacC[:zfin,yin:,:].shape)
    drF_exp = drF_exp + np.zeros(hFacC[:zfin,yin:,:].shape)
    
    ShelfVolume = hFacC[:zfin,yin:,:]*drF_exp*rA_exp
    ShelfVolume_exp = np.expand_dims(ShelfVolume,0)
    ShelfVolume_exp = ShelfVolume_exp + np.zeros(HighConc_Mask.shape)
    
    HighConc_CellVol = np.ma.masked_array(ShelfVolume_exp,mask = HighConc_Mask) 
    VolWaterHighConc = np.ma.sum(np.ma.sum(np.ma.sum(HighConc_CellVol,axis = 1),axis=1),axis=1)
    
    #Get total mass of tracer on shelf
    Total_Tracer = np.ma.sum(np.ma.sum(np.ma.sum(ShelfVolume_exp*TrMask[:,:zfin,yin:,:]*1000.0,axis = 1),axis=1),axis=1) 
    # 1 m^3 = 1000 l
    
    return (VolWaterHighConc, Total_Tracer)

#---------------------------------------------------------------------------------------------------------------------------
def howMuchWaterCV(Tr,MaskC,nzlim,rA,hFacC,drF,yin,zfin,xi,yi,xo,xf):
    '''
    INPUT----------------------------------------------------------------------------------------------------------------
    Tr    : Array with concentration values for a tracer. Until this function is more general, this should be size 19x90x360x360
    MaskC : Land mask for tracer
    nzlim : The nz index under which to look for water properties
    rA    : Area of cell faces at C points (360x360)
    fFacC : Fraction of open cell (90x360x360)
    drF   : Distance between cell faces (90)
    yin   : across-shore index of shelf break
    zfin  : shelf break index + 1 
    xi    : initial profile x index
    yi    : initial profile y index
    xo : initial x index of control volume
    xf : final x index of control volume
    OUTPUT----------------------------------------------------------------------------------------------------------------
    VolWaterHighConc =  Array with the volume of water over the shelf [:,:30,227:,:] at every time output.
    Total_Tracer =  Array with the mass of tracer (m^3*[C]*l/m^3) at each x-position over the shelf [:,:30,227:,:] at 
                    every time output. Total mass of tracer at xx on the shelf.
                                                
    -----------------------------------------------------------------------------------------------------------------------
    '''
    maskExp = maskExpand(MaskC,Tr)

    TrMask=np.ma.array(Tr,mask=maskExp)   
    
    trlim = TrMask[0,nzlim,yi,xi]
    
    print('tracer limit concentration is: ',trlim)
    
    WaterX = 0
    
    # mask cells with tracer concentration < trlim on control volume
    HighConc_Masked = np.ma.masked_less(TrMask[:,:zfin,yin:,xo:xf], trlim) 
    HighConc_Mask = HighConc_Masked.mask
    
    #Get volume of water of cells with relatively high concentration
    rA_exp = np.expand_dims(rA[yin:,xo:xf],0)
    drF_exp = np.expand_dims(np.expand_dims(drF[:zfin],1),1)
    rA_exp = rA_exp + np.zeros(hFacC[:zfin,yin:,xo:xf].shape)
    drF_exp = drF_exp + np.zeros(hFacC[:zfin,yin:,xo:xf].shape)
    
    ShelfVolume = hFacC[:zfin,yin:,xo:xf]*drF_exp*rA_exp
    ShelfVolume_exp = np.expand_dims(ShelfVolume,0)
    ShelfVolume_exp = ShelfVolume_exp + np.zeros(HighConc_Mask.shape)
    
    HighConc_CellVol = np.ma.masked_array(ShelfVolume_exp,mask = HighConc_Mask) 
    VolWaterHighConc = np.ma.sum(np.ma.sum(np.ma.sum(HighConc_CellVol,axis = 1),axis=1),axis=1)
    
    #Get total mass of tracer on shelf
    Total_Tracer = np.ma.sum(np.ma.sum(np.ma.sum(ShelfVolume_exp*TrMask[:,:zfin,yin:,xo:xf]*1000.0,axis = 1),axis=1),axis=1) 
    # 1 m^3 = 1000 l
    
    return (VolWaterHighConc, Total_Tracer)

#---------------------------------------------------------------------------------------------------------------------------

def dumpFiles(filename,variable,form = 'dump'):
    
    '''Filename is a string with the path,filename and extension to write into; variable is the np array to save 
    and form is the file format to save to, it can be either 'dump' which uses np.ma.dump or 'txt' for a regular 
    text file. To load the arrays use np.load(filename)'''
     
    if form == 'dump':
        np.ma.dump(variable,filename)
    elif form == 'txt':
        np.savetxt(filename, variable)
    else:
        print('Format has to be dump or txt')
        
    
#---------------------------------------------------------------------------------------------------------------------------





