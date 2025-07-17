#!/usr/bin/env python
import numpy as np
import xarray as xr
import gsw_xarray as gsw
from numba import njit

from .loading import mask_basin

@njit
def mocsig_j(dens, vel, dxu, dz, sigmas, mask=None):
    """
    Fast computation of density-space overturning streamfunction using numba.
    
    Args:
        dens (k,j,i): Potential density in kg/m^3 - 1000
        vel (k,j,i): Northward velocities in cm/s (native POP units)
        dxu (j,i): Width of northern cell edges in cm (native POP units)
        dz (k): Layer depths in m
        sigmas (sigma): List, range or numpy array of potential densities for the streamfunction
        mask (j,i): =1 over the basin and =0 elsewhere. (default: None, i.e., calculation will be global)
    NB! All inputs must be provided as numpy arrays with the shapes specified above. No time dimension is allowed.

    Returns:
        density-space streamfunction with dimensions (sigma, j)
    """
    ktot, jtot, itot = dens.shape # dens and vel have shapes (k, j, i)
    mocsig = np.zeros((len(sigmas), jtot-1), dtype=np.float64)
    if mask is None:
        mask_basin = np.ones((jtot, itot))
    else:
        mask_basin = mask
    
    for j in range(jtot-1):
        for k in range(ktot):
            for i in range(itot):
                if mask_basin[j,i]>0.:
                    v_transp = vel[k,j,i]*1e-2*dxu[j,i]*1e-2*dz[k] # v*dx*dz (m^3/s)
                    if i<itot-1:
                        sigma = (dens[k,j,i]+dens[k,j+1,i]+dens[k,j,i+1]+dens[k,j+1,i+1])/4
                    else:
                        sigma = (dens[k,j,i]+dens[k,j+1,i]+dens[k,j,0]+dens[k,j+1,0])/4
                    for s, sig in enumerate(sigmas):
                        if sigma>sig:
                            mocsig[s,j] -= v_transp*1e-6 # in Sv
    return mocsig

def mocsig_xr(pop_mon, grid, dz, p_level=0, sigmas=np.arange(23, 28.201, 0.1)):
    """
    Calculate density-space overturning streamfunction for xarray input/output. Uses `:mocsig_j`

    Args:
        pop_mon: POP output fields for one month as xarray.Dataset
        grid: POP grid file as xarray.Dataset
        dz: Layer depths in m as xarray.DataArray
        p_level: Reference depth for density in m (default: 0)
        sigmas:  List, range or numpy array of potential densities for the streamfunction
    
    Returns:
        xarray.Dataset with dimensions (basin, sigma, j)
    """
    # Atlantic + Arctic without Med
    mask_atl = mask_basin(grid, "atlantic_arctic").astype(np.float64).values
    # Atlantic + Nordic Seas w/o Med and w/o Arctic
    #mask_atl = ((grid["REGION_MASK"]==6) | (grid["REGION_MASK"]==8) | (grid["REGION_MASK"]==9)).astype(np.float64).values
    # Indo-Pacific
    mask_ip = mask_basin(grid, "indo_pacific").astype(np.float64).values
    mask_glob = mask_basin(grid, "global").astype(np.float64).values
    
    if p_level == 0 and "PD" in list(pop_mon.data_vars):
        # surface density (sigma_0) in POP output
        dens_in = (pop_mon["PD"]-1)*1000
    else:
        # Calculate density
        t_conservative = gsw.CT_from_pt(SA=pop_mon["SALT"]*1000, pt=pop_mon["TEMP"])
        dens_in = gsw.rho(SA=pop_mon["SALT"]*1000, CT=t_conservative, p=p_level)-1000
    
    mocsig_glob = mocsig_j(dens_in.values, pop_mon["VVEL"].values, grid["DXU"].values, dz.values, sigmas, mask=mask_glob)
    mocsig_atl = mocsig_j(dens_in.values, pop_mon["VVEL"].values, grid["DXU"].values, dz.values, sigmas, mask=mask_atl)
    mocsig_ip = mocsig_j(dens_in.values, pop_mon["VVEL"].values, grid["DXU"].values, dz.values, sigmas, mask=mask_ip)
    
    jtot = len(pop_mon.j)
    return xr.DataArray(
        np.asarray([mocsig_glob, mocsig_atl, mocsig_ip]),
        coords={
            "basin": ["global", "atlantic_arctic", "indo_pacific"],
            "sigma": sigmas, "j": range(jtot-1)
        }
    )

