# ShelfBreakTools - Find the shelf break indices; Get shelf break wall fields and plot those fields.

import matplotlib.pyplot as plt

import numpy as np

import scipy as spy

import pylab as pl

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def findShelfBreak(zlev,hfac):
    '''Find the x and y indices of the shelf break cells at a given vertical level. 
    This function looks for the first element of hfac[zlev,:,kk] (all the elements of 
    hfac at a certain zlevel and alongshore position) that is halfway closed (hfac<=0.5) 
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
    
    #SBIndx = np.empty(nx+2) # I have to add 2 extra points that the algorithm cannot find
    #SBIndy = np.empty(nx+2)  
    SBIndx = np.empty(nx) # No need to add them anymore
    SBIndy = np.empty(nx)

    for kk in range(nx):
        #SBIndy[kk] = np.argmax(hfac[zlev,:,kk]!=1) # use this for old grid
        SBIndy[kk] = np.argmax(hfac[zlev,:,kk] < 0.96)  # use this for quad grid
        SBIndx[kk] = kk

    #SBIndy[kk+1] = 216    # Since I changed the condition, I don't need these extra points anymore
    #SBIndy[kk+2] = 216
    
    #SBIndx[kk+1] = 149
    #SBIndx[kk+2] = 210
    
    SBx = SBIndx.astype(int)
    SBy = SBIndy.astype(int)
    
    ind = np.argsort(SBx)    # make sure they are correctly sorted in ascending order - this is important to calculate slopes later
    
    sortSBx = SBx[ind]
    sortSBy = SBy[ind]
    
    
    return(sortSBx,sortSBy)
    
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def MerFluxSB(SBxx,SByy,time,Flux,z,x,zlev,hfac,Mask):
    '''Flux across shelf break - This function uses findShelfBreak to get the indices of the cells along the shelf break 
    (with or without canyon) and returns a (nz,nx) array with the flux across those cells from bottom to surface. The flux 
    should be normal to the shelf break, so only the meridional component of flux is used. This function can also work 
    for transport.
     -------------------------------------------------------------------------------------------------------------------
     INPUT: 
	    SBxx,SByy - indices of SB from findShelfBreak function
	    time -  time output 
            Flux - array with meridional flux data from MITgcm model. The shape should be (nt,nz,ny,nx)
            z - 1D array with z-level depth data
            x - alongshore coordinates (2D)
            hfac - open cell fraction that works as mask
            zlev - vertical level at which to get the shelf break indices
    OUTPUT : array [nz,nx] with flux values across shelfbreak
    ----------------------------------------------------------------------------------------------------------------------
    '''
    
    SBxx, SByy = findShelfBreak(zlev,hfac)

    sizeX = np.shape(SBxx)
    sizes = np.shape(hfac)
    
    nx = sizeX[0]
    nz = sizes[0]
    
    FluxY = np.empty((nz,nx))
    MaskY = np.empty((nz,nx))

    unstagFlux = np.add(Flux[..., :-1, :], Flux[..., 1:, :]) / 2
    
    kk = 0
    for index in zip(SBxx,SByy):
        #print(index)
        FluxY[:,kk] = unstagFlux[time,:,index[1],index[0]] 
        MaskY[:,kk] = Mask[:,index[1],index[0]]
        kk = kk+1

    FluxYmask = np.ma.array(FluxY,mask=MaskY)
    return(FluxYmask)
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ZonFluxSB(SBxx,SByy,time,Flux,z,x,zlev,hfac,Mask):
    '''Zonal Flux across shelf break - This function uses fingShelfBreak to get the indices of the cells along the shelf break 
    (with or without canyon) and returns a (nz,nx) array with the flux across those cells from bottom to surface. This function can also work 
    for transport.
     ----------------------------------------------------------------------------------------------------------------------------
     INPUT: 
	    SBxx,SByy - indices od SB from findShelfBreak function
	    time -  time output 
            Flux - array with zonal flux data from MITgcm model. The shape should be (nt,nz,ny,nx)
            z - 1D array with z-level depth data
            x - alongshore coordinates (2D)
            hfac - open cell fraction that works as mask
            zlev - vertical level at which to get the shelf break indices
    OUTPUT : array [nz,nx] with flux values across shelfbreak
    -----------------------------------------------------------------------------------------------------------------------------
    '''
    sizeX = np.shape(SBxx)
    sizes = np.shape(hfac)
    
    nx = sizeX[0]
    nz = sizes[0]
    
    FluxX = np.empty((nz,nx))
    MaskX = np.empty((nz,nx))

    unstagFlux = np.add(Flux[..., :, :-1], Flux[..., :, 1:]) / 2
    
    kk = 0
    for index in zip(SBxx,SByy):
        #print(index)
        FluxX[:,kk] = unstagFlux[time,:,index[1],index[0]] 
        MaskX[:,kk] = Mask[:,index[1],index[0]]
        kk = kk+1

    FluxXmask = np.ma.array(FluxX,mask=MaskX)
    return(FluxXmask)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def MerFluxSBNoUnstag(SBxx,SByy,time,Flux,z,x,zlev,hfac,Mask):
    '''Flux across shelf break No Unstagger - This function uses fingShelfBreak to get the indices of the cells along the shelf break 
    (with or without canyon) and returns a (nz,nx) array with the flux across those cells from bottom to surface. The flux 
    should be normal to the shelf break, so only the meridional component of flux is used. The flux should be unstaggered into C grid. This function can also work 
    for transport.
     -------------------------------------------------------------------------------------------------------------------
     INPUT: 
	    SBxx,SByy - indices od SB from findShelfBreak function
	    time -  time output 
            Flux - array with meridional flux data from MITgcm model. The shape should be (nt,nz,ny,nx)
            z - 1D array with z-level depth data
            x - alongshore coordinates (2D)
            hfac - open cell fraction that works as mask
            zlev - vertical level at which to get the shelf break indices
    OUTPUT : array [nz,nx] with flux values across shelfbreak
    ----------------------------------------------------------------------------------------------------------------------
    '''
    
    SBxx, SByy = findShelfBreak(zlev,hfac)

    sizeX = np.shape(SBxx)
    sizes = np.shape(hfac)
    
    nx = sizeX[0]
    nz = sizes[0]
    
    FluxY = np.empty((nz,nx))
    MaskY = np.empty((nz,nx))

    kk = 0
    for index in zip(SBxx,SByy):
        #print(index)
        FluxY[:,kk] = Flux[time,:,index[1],index[0]] 
        MaskY[:,kk] = Mask[:,index[1],index[0]]
        kk = kk+1

    FluxYmask = np.ma.array(FluxY,mask=MaskY)
    return(FluxYmask)
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ZonFluxSBNoUnstag(SBxx,SByy,time,Flux,z,x,zlev,hfac,Mask):
    '''Zonal Flux across shelf break - This function uses fingShelfBreak to get the indices of the cells along the shelf break 
    (with or without canyon) and returns a (nz,nx) array with the flux across those cells from bottom to surface. The flux should be unstaggered into C grid. This function can also work 
    for transport.
     ----------------------------------------------------------------------------------------------------------------------------
     INPUT: 
	    SBxx,SByy - indices od SB from findShelfBreak function
	    time -  time output 
            Flux - array with zonal flux data from MITgcm model. The shape should be (nt,nz,ny,nx)
            z - 1D array with z-level depth data
            x - alongshore coordinates (2D)
            hfac - open cell fraction that works as mask
            zlev - vertical level at which to get the shelf break indices
    OUTPUT : array [nz,nx] with flux values across shelfbreak
    -----------------------------------------------------------------------------------------------------------------------------
    '''
    sizeX = np.shape(SBxx)
    sizes = np.shape(hfac)
    
    nx = sizeX[0]
    nz = sizes[0]
    
    FluxX = np.empty((nz,nx))
    MaskX = np.empty((nz,nx))

    
    kk = 0
    for index in zip(SBxx,SByy):
        #print(index)
        FluxX[:,kk] = Flux[time,:,index[1],index[0]] 
        MaskX[:,kk] = Mask[:,index[1],index[0]]
        kk = kk+1

    FluxXmask = np.ma.array(FluxX,mask=MaskX)
    return(FluxXmask)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fieldSB(time,field,z,x,zlev,hfac,Mask):
    '''selects subset of field to be only along the shelf break - This function uses findShelfBreak to get the indices of the cells along the shelf break 
    (with or without canyon) and returns a (nz,nx) array with the field across those cells from bottom to surface. The field should be on cell centers 
     -------------------------------------------------------------------------------------------------------------------
     INPUT: time -  time output 
            Field - array with some variable data from MITgcm model. The shape should be (nt,nz,ny,nx)
            z - 1D array with z-level depth data
            x - alongshore coordinates (2D)
            hfac - open cell fraction that works as mask
            zlev - vertical level at which to get the shelf break indices
    OUTPUT : array [nz,nx] with field values across shelfbreak
    ----------------------------------------------------------------------------------------------------------------------
    '''
    SBxx, SByy = findShelfBreak(zlev,hfac)

    sizeX = np.shape(SBxx)
    sizes = np.shape(hfac)
    
    nx = sizeX[0]
    nz = sizes[0]
    
    fieldY = np.empty((nz,nx))
    MaskY = np.empty((nz,nx))

    kk = 0
    for index in zip(SBxx,SByy):
        fieldY[:,kk] = field[time,:,index[1],index[0]] 
        MaskY[:,kk] = Mask[:,index[1],index[0]]
        kk = kk+1

    fieldYmask = np.ma.array(fieldY,mask=MaskY)
    return(fieldYmask)
    
    
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
    sizes = np.shape(FluxPlot)
    nx = sizes[1]
    
    plt.subplot(numRows,numCols,kk)
    ax = plt.gca()

    ax.set_axis_bgcolor((205/255.0, 201/255.0, 201/255.0))
    plt.contourf(x[1,4:360-5],z[nzmin:nzmax],FluxPlot[nzmin:nzmax,:],cmap = "RdYlBu_r")
    

    if abs(np.max(FluxPlot)) >= abs(np.min(FluxPlot)):
        pl.clim([-np.max(FluxPlot),np.max(FluxPlot)])
    else:
        pl.clim([np.min(FluxPlot),-np.min(FluxPlot)])
    
    
    plt.axvline(x=x[1,120],linestyle='-', color='0.75')
    plt.axvline(x=x[1,240],linestyle='-', color='0.75')
    
    plt.xlabel('Along-shelfbreak index')
        
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
    sizes = np.shape(FluxPlot)
    nx = sizes[1]
    
    plt.subplot(numRows,numCols,kk)
    ax = plt.gca()

    ax.set_axis_bgcolor((205/255.0, 201/255.0, 201/255.0))
    plt.pcolor(x[1,4:360-5],z[nzmin:nzmax],FluxPlot[nzmin:nzmax,:],cmap = "RdYlBu_r")

    if abs(np.max(FluxPlot)) >= abs(np.min(FluxPlot)):
        pl.clim([-np.max(FluxPlot),np.max(FluxPlot)])
    else:
        pl.clim([np.min(FluxPlot),-np.min(FluxPlot)])
    
    plt.axvline(x=x[1,120],linestyle='-', color='0.75')
    plt.axvline(x=x[1,240],linestyle='-', color='0.75')
    
   
    plt.ylabel('m')

    cb = plt.colorbar()

    cb.set_label(units,position=(1, 0),rotation=0)

    plt.title(" %1.1f days " % ((tt/2.)+0.5))

 
 # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def AreaXface(hfac,dr,dx,zlev):
    '''Calculate area of Shelf break wall across x axis - perpendicular to y component of vel.
    -----------------------------------------------------------------------------------
    INPUT
    hfac : Fraction of open cell at cell center (hFacC)     
    dr : r cell face separation (drf)
    dx : x cell center separation (dxf)
    zlev : vertical level to find shelf break indices
    
    NOTE - This function uses findShelfBreak(zlev,hfac) to get the x, y indices of shelf break.
    
    OUTPUT
    area : np 2D array size x,z 
    '''
    
    SBxx, SByy = findShelfBreak(zlev,hfac)

    sizes = np.shape(hfac)
    nx = sizes[2]
    ny = sizes[1]
    nz = sizes[0]
    
    area = np.empty((nz,nx))
    
    for ii in range(nx):
        area[:,ii] = hfac[:,SByy[ii],SBxx[ii]] * dr[:] * dx[SByy[ii],SBxx[ii]]
   
    return(area)

 # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def AreaYface(hfac,dr,dy,zlev):
    '''Calculate area of Shelf break wall across y axis - perpendicular to x component of vel.
    -----------------------------------------------------------------------------------
    INPUT
    hfac : Fraction of open cell at cell center (hFacC)     
    dr : r cell face separation (drf)
    dy : y cell center separation (dyf)
    zlev : vertical level to find shelf break indices
    
    NOTE - This function uses findShelfBreak(zlev,hfac) to get the x, y indices of shelf break.
    
    OUTPUT
    area : np 2D array size x,z 
    '''
    
    SBxx, SByy = findShelfBreak(zlev,hfac)

    sizes = np.shape(hfac)
    nx = sizes[2]
    ny = sizes[1]
    nz = sizes[0]
    
    area = np.empty((nz,nx))
    
    for ii in range(nx):
        area[:,ii] = hfac[:,SByy[ii],SBxx[ii]] * dr[:] * dy[SByy[ii],SBxx[ii]]
   
    return(area)
   
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def findSlope(x,y,SBx,SBy):
  '''Calculate the slope and associated angle at every point along the shelf break. The algotithm calculates the slope y2-y1/x2-x1
  taking 9 points.This is to smooth out the spikes I get due to the steppiness of the shelf break. 
  NOTE: The final array is 8 points shorter than SBx.
  ----------------------------------------------------------------------------------------------------------
  INPUT
  x   : x coordinate at cell center, XC form MITgcm grid output 2D array (yindex, xindex).
  y   : y coordinate at cell center, YC form MITgcm grid output 2D array (yindex, xindex).
  SBx : x indices of shelfbreak cells. Output from findShelfBreak()
  SBy : y indices of shelfbreak cells. Output from findShelfBreak()
  
  OUTPUT
  slope : 1D array with the values of the slopes calculated 
  angle : arctan(slope) 
  '''
   
  deltaX = x[SBy[8:],SBx[8:]]-x[SBy[:-8],SBx[:-8]] # x2-x1
  deltaY = y[SBy[8:],SBx[8:]]-y[SBy[:-8],SBx[:-8]] # y2-y1
  
  slope = deltaY/deltaX
  
  theta = np.arctan(slope)
  
  return(slope, theta)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
