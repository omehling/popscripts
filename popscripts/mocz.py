#!/usr/bin/env python
import numpy as np
import xarray as xr

from .loading import mask_basin


def mocz_j(vel, dxu, dz, mask=None, remove_mean=False):
    """
    Computation of the depth-space overturning streamfunction for one basin.
    
    Args:
        vel (k,j,i): Northward velocities in cm/s (native POP units)
        dxu (j,i): Width of northern cell edges in cm (native POP units)
        dz (k): Layer depths in m
        mask (j,i): =1 over the basin and =0 elsewhere. (default: None, i.e., calculation will be global)
        remove_mean (bool): Remove section mean velocity before calculating MOC
    NB! All inputs must be provided as xarray Datasets. A time dimension is allowed.

    Returns:
        depth-space streamfunction with dimensions (lev, j)
    """
    vel_sel = vel.where(mask,0)
    if remove_mean:
        vel_mean = vel_sel.weighted(dxu*dz).mean(["i","k"])
        vel_sel = vel_sel-vel_mean
    
    moc = ((vel_sel*dxu*dz).sum("i").cumsum("k")*1e-10).rename({"k": "lev"})
    moc["lev"] = np.cumsum(dz).values
    moc.name = "msftyz"

    return moc

def mocz_xr(pop_mon, grid, dz, remove_mean=False):
    """
    Computation of the depth-space overturning streamfunction for multiple basins.

    Args:
        pop_mon: POP output fields for one month as xarray.Dataset
        grid: POP grid file as xarray.Dataset
        dz: Layer depths in m as xarray.DataArray
    
    Returns:
        xarray.Dataset with dimensions (basin, lev, j)
    """
    # Atlantic + Arctic without Med
    mask_atl = mask_basin(grid, "atlantic_arctic")
    mask_ip = mask_basin(grid, "indo_pacific")
    mask_glob = mask_basin(grid, "global")
    
    moc_glob = mocz_j(pop_mon["VVEL"], grid["DXU"], dz, mask=mask_glob, remove_mean=remove_mean).expand_dims({"basin": ["global"]})
    moc_atl = mocz_j(pop_mon["VVEL"], grid["DXU"], dz, mask=mask_atl, remove_mean=remove_mean).expand_dims({"basin": ["atlantic_arctic"]})
    moc_ip = mocz_j(pop_mon["VVEL"], grid["DXU"], dz, mask=mask_ip, remove_mean=remove_mean).expand_dims({"basin": ["indo_pacific"]})
    
    return xr.concat([moc_glob, moc_atl, moc_ip], dim="basin")

