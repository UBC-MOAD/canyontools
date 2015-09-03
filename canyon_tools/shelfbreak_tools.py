# ShelfBreakTools - Find the shelf break indices; Get shelf break wall fields and plot those fields.

import matplotlib.pyplot as plt

import numpy as np

import scipy as spy

import pylab as pl

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def findShelfBreak(zlev,hfac):
    '''Find the x and y indices of the shelf break cells at a given vertical level. 
    This function looks for the first element of hfac[zlev,:,kk] (all the elements of 
    hfac at a certain zlevel and alongshore position) that is completely closed (hfac=0) 
    and saves its x,y indices in the integer arrays SBx, SBy.
    -----------------------------------------------------------------------------------
    INPUT
         
    zlev : vertical level to find shelf break indices
    hfac : open-cell fraction array. It should be the cell-centered HFacC.
     
    OUTPUT
    
    SBx, SBy : two integer arrays containing the x and y indices (respectively) of the shelf break. 
    '''
    
    sizes = np.shape(hfac)
    nx = sizes[2]
    ny = sizes[1]
    
      
    SBIndx = np.empty(nx)
    SBIndy = np.empty(nx)

    for kk in range(nx):
        SBIndy[kk] = np.argmax(hfac[zlev,:,kk]==0)
        SBIndx[kk] = kk

    SBx = SBIndx.astype(int)
    SBy = SBIndy.astype(int)
    
    return(SBx,SBy)
    
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def FluxSB(time,Flux,z,x,zlev,hfac,Mask):
    '''Flux across shelf break - This function uses fingShelfBreak to get the indices of the cells along the shelf break 
    (with or without canyon) and returns a (nz,nx) array with the flux across those cells from bottom to surface. The flux 
    should be normal to the shelf break, so only the meridional component of flux is used. This function can also work 
    for transport.
     -------------------------------------------------------------------------------------------------------------------
     INPUT: time -  time output 
            Flux - array with meridional flux data from MITgcm model. The shape should be (nt,nz,ny,nx)
            z - 1D array with z-level depth data
            x - alongshore coordinates (2D)
            hfac - open cell fraction that works as mask
            zlev - vertical level at which to get the shelf break indices
    OUTPUT : array [nz,nx] with flux values across shelfbreak
    ----------------------------------------------------------------------------------------------------------------------
    '''
    SBxx, SByy = findShelfBreak(zlev,hfac)

    sizes = np.shape(hfac)
    nx = sizes[2]
    ny = sizes[1]
    nz = sizes[0]
    
    FluxY = np.empty((nz,nx))
    MaskY = np.empty((nz,nx))

    unstagFlux = np.add(Flux[..., :-1, :], Flux[..., 1:, :]) / 2
        
    for ii in range(nx):
        FluxY[:,ii] = unstagFlux[time,:,SByy[ii],SBxx[ii]] 
        MaskY[:,ii] = Mask[:,SByy[ii],SBxx[ii]]

    FluxYmask = np.ma.array(FluxY,mask=MaskY)
    return(FluxYmask)
    

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def contourfFluxSB(time,numCols,numRows,FluxPlot,z,x,units, nzmin,nzmax,kk,zlev):
    ''' Contourf plot of flux across shelf break wall' 
     -------------------------------------------------------------------------------------------------------------------
     INPUT:  time - timeslice at what we want to plot (integer, usually between 0 and 18)
            numCols, numRows - integers indicating, respectively, the number of columns and rows to arrange the subplots into.
             Flux - array with across shelf break flux data (nz,nx)
                z - 1D array with z-level depth data
                x - alongshore coordinates (2D)
            nzmin - integer indicating index of lower depth (z) to plot. Remember z goes form 0 to -2000 m. 
            nzmax - integer indicating index of upper depth (z) to plot. Remember z goes form 0 to -2000 m. 
            units - string with units for colorbar. E.g. units = '$molC\ m^{-1}\cdot m^3s^{-1}$' 
               kk - Integer inidcating the number of subplot 
             zlev - vertical level at which to get the shelf break indices
    OUTPUT : Nice contourf subplot
    ----------------------------------------------------------------------------------------------------------------------
    '''
    
    plt.subplot(numRows,numCols,kk)
    ax = plt.gca()

    ax.set_axis_bgcolor((205/255.0, 201/255.0, 201/255.0))
    plt.contourf(x[200,:],z[nzmin:nzmax],FluxPlot[nzmin:nzmax,:],cmap = "RdYlBu_r")

    if abs(np.max(FluxPlot)) >= abs(np.min(FluxPlot)):
        pl.clim([-np.max(FluxPlot),np.max(FluxPlot)])
    else:
        pl.clim([np.min(FluxPlot),-np.min(FluxPlot)])
    
    plt.axvline(x=x[0,130],linestyle='-', color='0.75')
    plt.axvline(x=x[0,-130],linestyle='-', color='0.75')
    
    plt.xlabel('m')
        
    plt.ylabel('m')

    cb = plt.colorbar()

    cb.set_label(units,position=(1, 0),rotation=0)

    plt.title(" %1.1f days " % ((time/2.)+0.5))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
def pcolorFluxSB(time,numCols,numRows,FluxPlot,z,x,units, nzmin,nzmax,kk,zlev):
    ''' pcolor plot of flux across shelf break wall' 
     -------------------------------------------------------------------------------------------------------------------
    INPUT:  time - timeslice at what we want to plot (integer, usually between 0 and 18)
            numCols, numRows - integers indicating, respectively, the number of columns and rows to arrange the subplots into.
             Flux - array with across shelf break flux data (nz,nx)
                z - 1D array with z-level depth data
                x - alongshore coordinates (2D)
            nzmin - integer indicating index of lower depth (z) to plot. Remember z goes form 0 to -2000 m. 
            nzmax - integer indicating index of upper depth (z) to plot. Remember z goes form 0 to -2000 m. 
            units - string with units for colorbar. E.g. units = '$molC\ m^{-1}\cdot m^3s^{-1}$' 
               kk - Integer inidcating the number of subplot 
             zlev - vertical level at which to get the shelf break indices
    OUTPUT : Nice pcolor subplot
    ----------------------------------------------------------------------------------------------------------------------
    '''
    plt.subplot(numRows,numCols,kk)
    ax = plt.gca()

    ax.set_axis_bgcolor((205/255.0, 201/255.0, 201/255.0))
    plt.pcolor(x[200,:],z[nzmin:nzmax],FluxPlot[nzmin:nzmax,:],cmap = "RdYlBu_r")

    if abs(np.max(FluxPlot)) >= abs(np.min(FluxPlot)):
        pl.clim([-np.max(FluxPlot),np.max(FluxPlot)])
    else:
        pl.clim([np.min(FluxPlot),-np.min(FluxPlot)])
    
    plt.axvline(x=x[0,130],linestyle='-', color='0.75')
    plt.axvline(x=x[0,-130],linestyle='-', color='0.75')
    
    plt.xlabel('m')
        
    plt.ylabel('m')

    cb = plt.colorbar()

    cb.set_label(units,position=(1, 0),rotation=0)

    plt.title(" %1.1f days " % ((tt/2.)+0.5))
