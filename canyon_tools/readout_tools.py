# ReadOutput tools MITgcm

import matplotlib.pyplot as plt

from math import *

import numpy as np

import os

from scipy.stats import nanmean

from netCDF4 import Dataset


def getField(statefile, fieldname):
    ''' Get field from MITgcm netCDF output. Field mut be at leat 2-D.
    :statefile : string with /path/to/state.0000000000.t001.nc
    :fieldname : string with the variable name as written on the netCDF file ('Temp', 'S','Eta', etc.)'''
    StateOut = Dataset(statefile)
    
    Fld = StateOut.variables[fieldname][:]
    
    shFld = np.shape(Fld)
    
        
    if len(shFld) == 2:
        
        Fld2 = np.reshape(Fld,(shFld[0],shFld[1])) # reshape to pcolor order
        return Fld2 
    
    elif len(shFld) == 3:
        
        Fld2 = np.zeros((shFld[0],shFld[1],shFld[2])) 
        Fld2 = np.reshape(Fld,(shFld[0],shFld[1],shFld[2])) # reshape to pcolor order
        return Fld2
        
    elif len(shFld) == 4:
        
        Fld2 = np.zeros((shFld[0],shFld[1],shFld[2],shFld[3])) 
        Fld2 = np.reshape(Fld,(shFld[0],shFld[1],shFld[2],shFld[3])) # reshape to pcolor order
        return Fld2
        
    else:
        
        print (' Check size of field ')
    
def unstagger(ugrid, vgrid):
    """ Interpolate u and v component values to values at grid cell centres (from D.Latornell for NEMO output).

    The shapes of the returned arrays are 1 less than those of
    the input arrays in the y and x dimensions.

    :arg ugrid: u velocity component values with axes (..., y, x)
    :type ugrid: :py:class:`numpy.ndarray`

    :arg vgrid: v velocity component values with axes (..., y, x)
    :type vgrid: :py:class:`numpy.ndarray`

    :returns u, v: u and v component values at grid cell centres
    :rtype: 2-tuple of :py:class:`numpy.ndarray`
    """
    u = np.add(ugrid[..., :-1], ugrid[..., 1:]) / 2
    v = np.add(vgrid[..., :-1, :], vgrid[..., 1:, :]) / 2
    #return u[..., 1:, :], v[..., 1:]
    return u, v


def getMask(GridFile, CellType):
    ''' Get cell-center, u-cell or v-cell mask
     gridfile: string containing NC grid filename
     CellType: String with HFac field name. It can be 'HFacC' for cell-center, 'HFacW' for open-side cell
     or 'HFacS' for other cell ?'''
  
    hFac = getField(GridFile,CellType) 

    hFacmasked = np.ma.masked_values(hFac, 0)

    MASKhFac = np.ma.getmask(hFacmasked)
    
    return MASKhFac

def calc_sigmaHor(RhoRef,T,S, At = 2.0E-4, Bs = 7.4E-4):
    '''Calculate sigma as sigma = sigma0 + (RhoRef[Bs(S-S0) - At(T-T0)]) with sigma0 = 0, T0 = 0 and S0 = 0.
       RhoRef: Reference salinity at model layer nz, matching z of T and S.
       T : 2D Temp field (nx,ny) at z
       S : 2D Salt field (nx,ny) at z
       At: Thermal expansion coefficient (units K^-1)
       Bs: Haline expansion coefficient (units ppt^-1)
       
       returns sigma : 2D density anomaly 
    '''
    
    sigma = RhoRef*(Bs*S - At*T)
    return sigma
    
def calc_sigmaVer(RhoRef,T,S, At = 2.0E-4, Bs = 7.4E-4):
    '''Calculate sigma as sigma = sigma0 + (RhoRef[Bs(S-S0) - At(T-T0)]) with sigma0 = 0, T0 = 0 and S0 = 0.
       RhoRef: Reference density profile at nx,ny.
       T : 1D Temp field at nx,ny 
       S : 1D Salt field at nx,ny
       At: Thermal expansion coefficient (units K^-1)
       Bs: Haline expansion coefficient (units ppt^-1)
       
       returns sigma : 1D density anomaly profile
    '''
    
    sigma = RhoRef*(Bs*S - At*T)
    return sigma
    
