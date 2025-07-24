#!/usr/bin/env python
from numba import njit
import numpy as np
import xarray as xr
import gsw_xarray as gsw

from .loading import load_straits

@njit
def transport_zonal(i, j, k, VVEL, DXU, dz):
    return 0.5*(VVEL[k,j,i]*DXU[j,i]*dz[k]
               +VVEL[k,j,i-1]*DXU[j,i-1]*dz[k])

@njit
def tracer_zonal(i, j, k, TRACER):
    return 0.5*(TRACER[k,j+1,i] + TRACER[k,j,i])

@njit
def transport_meridional(i, j, k, UVEL, DYU, dz):
    return 0.5*(UVEL[k,j,i]*DYU[j,i]*dz[k]
               +UVEL[k,j-1,i]*DYU[j-1,i]*dz[k])

@njit
def tracer_meridional(i, j, k, TRACER):
    return 0.5*(TRACER[k,j,i+1] + TRACER[k,j,i])

@njit
def strait_transport_zonal(j_strait, i_strait_start, i_strait_stop, VVEL, DXU, dz):
    transport_total = 0.

    for k in range(len(dz)):
        for i in range(i_strait_start-1, i_strait_stop):
            transport_total += transport_zonal(i, j_strait-1, k, VVEL, DXU, dz)

    return transport_total*1e-12  # in Sv

@njit
def strait_transport_meridional(i_strait, j_strait_start, j_strait_stop, UVEL, DYU, dz):
    transport_total = 0.

    for k in range(len(dz)):
        for j in range(j_strait_start-1, j_strait_stop):
            transport_total += transport_meridional(i_strait-1, j, k, UVEL, DYU, dz)

    return transport_total*1e-12  # in Sv

@njit
def density_overturning_zonal(j_strait, i_strait_start, i_strait_stop,
                              density_range, VVEL, PD, DXU, dz):
    density_overturning = np.zeros(len(density_range))
    for k in range(len(dz)):
        for i in range(i_strait_start-1, i_strait_stop):
            transport_local = transport_zonal(i, j_strait-1, k, VVEL, DXU, dz)
            dens_local = tracer_zonal(i, j_strait-1, k, PD)
            d = 0
            while d < len(density_range) and dens_local > density_range[d]:
                density_overturning[d] += transport_local
                d+=1
    return density_overturning*1e-12*-1

@njit
def density_overturning_meridional(i_strait, j_strait_start, j_strait_stop,
                                   density_range, UVEL, PD, DYU, dz):
    density_overturning = np.zeros(len(density_range))
    for k in range(len(dz)):
        for j in range(j_strait_start-1, j_strait_stop):
            transport_local = transport_meridional(i_strait-1, j, k, UVEL, DYU, dz)
            dens_local = tracer_meridional(i_strait-1, j, k, PD)
            d = 0
            while d < len(density_range) and dens_local > density_range[d]:
                density_overturning[d] += transport_local
                d+=1
    return density_overturning*1e-12*-1


def mocsig_strait(pop_mon, res, grid, dz, p_level=0, sigmas=np.arange(23, 28.201, 0.1), straits=None):
    """
    Density-space overturning streamfunction across different gateways.
    So far only supports zonal or meridional (no diagonal) sections.
    Reads in a list of straits from a file in `loading.load_straits`, see there for more info.
    
    Args:
        pop_mon: POP output fields for one month as xarray.Dataset
        res: Model resolution (gx1v6 or tx0.1v2)
        grid: POP grid file as xarray.Dataset
        dz: Layer depths in m as xarray.DataArray
        p_level: Reference depth for density in m (default: 0)
        sigmas:  List, range or numpy array of potential densities for the streamfunction
        straits: Only calculate streamfunctions for this subset of straits
                 (default: None = include all straits)
    
    Returns:
        xarray.Dataset with dimensions (strait, sigma)

    """
    strait_list_sel = load_straits(res, subset=straits)

    if p_level == 0 and "PD" in list(pop_mon.data_vars):
        # surface density (sigma_0) in POP output
        dens_in = (pop_mon["PD"]-1)*1000
    else:
        # Calculate density
        t_conservative = gsw.CT_from_pt(SA=pop_mon["SALT"]*1000, pt=pop_mon["TEMP"])
        dens_in = gsw.rho(SA=pop_mon["SALT"]*1000, CT=t_conservative, p=p_level)-1000

    v_load = pop_mon["VVEL"].values
    u_load = pop_mon["UVEL"].values

    dens_overturning = []

    for _, strait in strait_list_sel.iterrows():
        if strait["orientation"] == "zonal":
            overturning_strait = density_overturning_zonal(
                strait["j_start"], strait["i_start"], strait["i_stop"],
                sigmas, v_load, dens_in, grid["DXU"], dz
            )
        else:
            overturning_strait = density_overturning_meridional(
                strait["i_start"], strait["j_start"], strait["j_stop"],
                sigmas, u_load, dens_in, grid["DYU"], dz
            )
        overturning_strait_xr = xr.DataArray(overturning_strait, coords={"sigma": sigmas})
        dens_overturning.append(overturning_strait_xr)

    dens_overturning_xr = xr.concat(dens_overturning, dim="strait")
    dens_overturning_xr["strait"] = strait_list_sel["strait"]

    return dens_overturning_xr