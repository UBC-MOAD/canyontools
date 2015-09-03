
import matplotlib.pyplot as plt

from math import *

import numpy as np

import os

from scipy.stats import nanmean


def make_arbitrary_topo_smooth(total_fluid_depth,cR,W,Wsb,L,p,x,x_wall,y,y_base,y_bc,y_sb,y_coast,z_bottom,z_bc,z_sb,z_wall):
  """This function was originally written for python, then translated to matlab. I took the matlab version form Jessica Spurgin's files for MITgcm:
  This is a function that will return a depth field (topography) with a sech-shaped canyon.
  input: 
              total_fluid_depth = the depth of the fluid or the total depth
	      y_wall = the length of the ocean basin (y-axis)(going from 0=deep ocean m high values = coast
               x_wall = the width of the ocean basin (x-axis)
	      y_base = the distance where the slope begins to rise from the bottom
		    y_bc = the distance to the canyon mouth
		    y_sb = the distance of the shelf break
		  y_100 = the distance to the 100m isobath
		    y_50 = the distance to the 50m isobath
		    y_coast = the distance to the "coast" where topography stops increasing, but 	depth NOT =0
	      z_bottom = depth of the "deep ocean"; z is measured from the bottom up
	      z_bc = depth of the 1000 m contour
			z_800 = depth of the 800 m contour
			z_600 = depth of the 600 m contour
			z_400 = depth of the 400 m contour
			z_200 = depth of the 200 m contour
		    z_sb = shelf break depth (150m)
		  z_100 = depth of the 100 m contour
			  z50 = depth of the 50 m contour
		  z_wall = the depth of the topography beyond the y_coast variable (i.e. z_wall NOT = total fluid depth)
		      cR = the radius of curvature at the shelf break depth
		      W = the width at half the length at the shelf break depth
		    Wsb = the width at the shelf break
		      L = the length of the canyon
		    p,q = geometric parameters used to help shape the canyon see geometry.ods	    """
  
  # Slope profile is the topography without the canyon
  slope_profile = tanktopo(total_fluid_depth,y,y_base,y_bc,y_sb,y_coast,z_bottom,z_bc,z_sb,z_wall)
  
  # Canyon Profile defines the slope of the canyon
  canyon_profile = canyontopo(total_fluid_depth,L,y,y_sb,y_coast,z_sb,z_wall)
  
  # Width profile defines the slope of the canyon as well as the shape
  width_profile = widthprofile(cR,W,Wsb,L,p,y,y_base,y_sb)
  
  # finding the depth of the canyon and setting negative values to zero
  canyondepth = slope_profile - canyon_profile
  
  canyondepth[canyondepth < 0] = 0
  
  # putting everything together to get the topography of the tank with a sech shaped canyon
  topography = np.zeros((len(y),len(x)))
  
  for j in np.arange(len(x)):
    # the width profile here is in m but in the python code it is in cm and
    # the coefficient is 45.5 instead of 0.455 I changed it so the units would work
    topography[:,j] = (slope_profile-canyondepth*(1.0/(np.cosh(0.455/width_profile*(x[j]-(0.5*x_wall))))**50))
   
  # Transposing the topography so that it is the same order as fortran reads
  # If this is not done the grid will not look right when implemented into the gcm
  topography=np.transpose(topography)
   
  return topography


def tanktopo(total_fluid_depth,y,y_base,y_bc,y_sb,y_coast,z_bottom,z_bc,z_sb,z_wall):
  """ Finds the topography in the tank without the canyon
  finds points at different parts of the topography using the equation of the liney=mx+b
  where topo_sp = y = mx+yo-mxo the values in the y & z are the corresponding bathymetry lines"""
  # The values below are specific to the outer canyon slope & used only here
  
  ys_200 = 49770 #y_wall-(0.44956*y_wall);
  ys_400 = 41100 #y_wall-(0.64222*y_wall);
  ys_600 = 35900 #y_wall-(0.75778*y_wall);
  
  #The slopes indicates the slope from the point below to the point indictaed
  #in the value name (i.e. sls_50 is the slope from ys_tops to ys_50, etc...
   
  sls_coast = 0.003
  sls_50 = 0.00618
  sls_100 = 0.01018
  sls_sb = 0.034
  sls_bc = 0.2439
   
  topo_sp = np.zeros(len(y))
  slope_profile = np.zeros(len(y))
   
  for jj in np.arange(len(y)):
     
    if y[jj] > y_base and y[jj]<= y_bc:
      topo_sp[jj] = sls_bc*y[jj]+ z_bottom - sls_bc*y_base
                    
    elif y[jj]> y_bc and y[jj]<= y_sb:
      topo_sp[jj] = sls_sb*y[jj] + z_bc - sls_sb*y_bc
                   
    elif y[jj]> y_sb and y[jj] < y_coast:
      topo_sp[jj] = 0.00614*y[jj] + z_sb - 0.00614*y_sb
                                  
    elif y[jj]>= y_coast:
      topo_sp[jj] = z_wall
                
  # subtract total fluid depth
    slope_profile[jj] = topo_sp[jj]-total_fluid_depth
  
  return slope_profile
   
def canyontopo(total_fluid_depth,L,y,y_sb,y_coast,z_sb,z_wall):
  """ define shelf/slope radii
        creates the slope of inside the canyon
        the input variables can be seen in the main function description"""
        
  topo_cp = np.zeros(len(y))
  canyon_profile = np.zeros(len(y))
    
  y_L = y_sb+L 
  yc_1100 = 51220
  yc_1000 = 54030
  yc_1200 = 40700 #48650 NWcanyon #40700 Barkley, 50500 Jessica's
    
  for ii in np.arange(len(y)):
    
    if y[ii] <= yc_1200:
      topo_cp[ii] = 0
           
    elif y[ii] > yc_1200 and y[ii]<= y_L: #yc_1000:
      topo_cp[ii] = 0.0621301775*y[ii] + 0 - 0.0621301775*yc_1200
                    
    elif y[ii] > y_L and y[ii] < y_coast :      
      topo_cp[ii] = 0.026*y[ii] + z_sb - 0.026*y_L #0.26
          
    elif y[ii] >= y_coast:
      topo_cp[ii] = z_wall
        
    # subtract total fluid depth
    canyon_profile[ii] = topo_cp[ii]-total_fluid_depth
 
  return canyon_profile

def widthprofile(cR,W,Wsb,L,p,y,y_base,y_sb):
  """ define shelf/slope radii (should be same as canyontopo above)
     define shape of topography (in m)
     the input variables can be seen above in the main function description"""
    
  # Set values for the width profile
  # calculate derived numbers to help shape the canyon
  
  sigmaa = 1.0/((9e-7)*cR)
  half = -Wsb/2.0+W/2.0 # width halfway between W and Wsb
  e = (L/2.0 -sigmaa*half**2)/half**p
    
  # scale factor
  sc = 1
    
  # fudge down to the deep
  alphaa = y_sb/y_base
    
  #The following sets of values correspond to width measurements from figure
  #1 of Flexxas et al., 2008
  #The 15 to 1 values indicate the approximate distance from the canyon head
  #(in km)
  #y90 is slightly less than the distance 10% from the canyon mouth 
    
  #This value is the newer (smaller) width at the canyon mouth; used for
  #smoothing the overall canyon width
    
  y90 = y_sb+0.062423*L
  W90 = Wsb-0.06651*Wsb
  dG_dx90 = p*e*(W90 - Wsb/2)**(p - 1) + 2*sigmaa*(W90 - Wsb)
  dW90 = 0.5/dG_dx90/sc
  A90 = (alphaa*Wsb-W90)/(y_base-y90)**2 - dW90/(y_base-y90)
    
  Wh = Wsb-0.9705*Wsb
  dG_dxh = p*e*(Wh - Wsb/2)**(p - 1) + 2*sigmaa*(Wh - Wsb)
  dh = 0.5/dG_dxh/sc
  Ah = (alphaa*Wsb-Wh)/(y_base-(y_sb+L))**2; #- dh/(y_base-(y_sb+L))
    
  wp = np.zeros(len(y))
    
  for l in np.arange(len(y)):
    
    if y[l] <= y_base:
      
      wp[l] = Wsb*alphaa + 9000
            
    elif y[l] > y_base and y[l] <= y_sb+L:
          
      wp[l] = Ah*(y[l]-(y_sb+L))**2 + dh*(y[l]-(y_sb+L))+Wh + 9000
               
    elif y[l] >= y_sb + L: # greater than canyon head
          
      wp[l] = 9390
        
  width_profile = wp  
     #width profile works smoothest with Y_base -> y_sb+L (using "head" terms)
     #-> >y_sb+L
   
  return width_profile
       
def make_two_canyons_smooth(total_fluid_depth,cR,W,Wsb,L,p,x,x_wall,y,y_base,y_bc,y_sb,y_coast,z_bottom,z_bc,z_sb,z_wall):
  """This function was originally written for python, then translated to matlab. I took the matlab version form Jessica Spurgin's files for MITgcm:
  This is a function that will return a depth field (topography) with a sech-shaped canyon.
  input: 
              total_fluid_depth = the depth of the fluid or the total depth
	      y_wall = the length of the ocean basin (y-axis)(going from 0=deep ocean m high values = coast
               x_wall = the width of the ocean basin (x-axis)
	      y_base = the distance where the slope begins to rise from the bottom
		    y_bc = the distance to the canyon mouth
		    y_sb = the distance of the shelf break
		  y_100 = the distance to the 100m isobath
		    y_50 = the distance to the 50m isobath
		    y_coast = the distance to the "coast" where topography stops increasing, but 	depth NOT =0
	      z_bottom = depth of the "deep ocean"; z is measured from the bottom up
	      z_bc = depth of the 1000 m contour
			z_800 = depth of the 800 m contour
			z_600 = depth of the 600 m contour
			z_400 = depth of the 400 m contour
			z_200 = depth of the 200 m contour
		    z_sb = shelf break depth (150m)
		  z_100 = depth of the 100 m contour
			  z50 = depth of the 50 m contour
		  z_wall = the depth of the topography beyond the y_coast variable (i.e. z_wall NOT = total fluid depth)
		      cR = the radius of curvature at the shelf break depth
		      W = the width at half the length at the shelf break depth
		    Wsb = the width at the shelf break
		      L = the length of the canyon
		    p,q = geometric parameters used to help shape the canyon see geometry.ods	    """
  
  # Slope profile is the topography without the canyon
  slope_profile = tanktopo(total_fluid_depth,y,y_base,y_bc,y_sb,y_coast,z_bottom,z_bc,z_sb,z_wall)
  
  # Canyon Profile defines the slope of the canyon
  canyon_profile = canyontopo(total_fluid_depth,L,y,y_sb,y_coast,z_sb,z_wall)
  
  # Width profile defines the slope of the canyon as well as the shape
  width_profile = widthprofile(cR,W,Wsb,L,p,y,y_base,y_sb)
  
  
  # finding the depth of the canyon and setting negative values to zero
  canyondepth = slope_profile - canyon_profile
  
  canyondepth[canyondepth < 0] = 0
  
  # putting everything together to get the topography of the tank with a sech shaped canyon
  topography = np.zeros((len(y),len(x)))
  
  for j in np.arange(len(x)):
    # the width profile here is in m but in the python code it is in cm and
    # the coefficient is 45.5 instead of 0.455 I changed it so the units would work
    topography[:,j] = (slope_profile-canyondepth*(1.0/(np.cosh(0.455/width_profile*(x[j]-(0.3*x_wall))))**50)-canyondepth*(1.0/(np.cosh(0.455/width_profile*(x[j]-(0.7*x_wall))))**50))
   
 
  # Transposing the topography so that it is the same order as fortran reads
  # If this is not done the grid will not look right when implemented into the gcm
  topography=np.transpose(topography)
   
  return topography

def make_flat_shelf(total_fluid_depth,cR,W,Wsb,L,p,x,x_wall,y,y_base,y_bc,y_sb,y_coast,z_bottom,z_bc,z_sb,z_wall):
  """This function was originally written for python, then translated to matlab. I took the matlab version form Jessica Spurgin's files for MITgcm:
  This is a function that will return a depth field (topography) of a shelf without a canyon.
  input: 
              total_fluid_depth = the depth of the fluid or the total depth
	      y_wall = the length of the ocean basin (y-axis)(going from 0=deep ocean m high values = coast
               x_wall = the width of the ocean basin (x-axis)
	      y_base = the distance where the slope begins to rise from the bottom
		    y_bc = the distance to the canyon mouth
		    y_sb = the distance of the shelf break
		  y_100 = the distance to the 100m isobath
		    y_50 = the distance to the 50m isobath
		    y_coast = the distance to the "coast" where topography stops increasing, but 	depth NOT =0
	      z_bottom = depth of the "deep ocean"; z is measured from the bottom up
	      z_bc = depth of the 1000 m contour
			z_800 = depth of the 800 m contour
			z_600 = depth of the 600 m contour
			z_400 = depth of the 400 m contour
			z_200 = depth of the 200 m contour
		    z_sb = shelf break depth (150m)
		  z_100 = depth of the 100 m contour
			  z50 = depth of the 50 m contour
		  z_wall = the depth of the topography beyond the y_coast variable (i.e. z_wall NOT = total fluid depth)
		      cR = the radius of curvature at the shelf break depth
		      W = the width at half the length at the shelf break depth
		    Wsb = the width at the shelf break
		      L = the length of the canyon
		    p,q = geometric parameters used to help shape the canyon see geometry.ods	    """
  
  # Slope profile is the topography without the canyon
  slope_profile = tanktopo(total_fluid_depth,y,y_base,y_bc,y_sb,y_coast,z_bottom,z_bc,z_sb,z_wall)
  
  topography = np.zeros((len(y),len(x)))
  
  for j in np.arange(len(x)):
    # the width profile here is in m but in the python code it is in cm and
    # the coefficient is 45.5 instead of 0.455 I changed it so the units would work
    topography[:,j] = (slope_profile)
   
  # Transposing the topography so that it is the same order as fortran reads
  # If this is not done the grid will not look right when implemented into the gcm
  topography=np.transpose(topography)
   
  return topography

