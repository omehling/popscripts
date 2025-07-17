#!/usr/bin/env python
import xarray as xr
import numpy as np

def check_res(res):
    if res not in ('gx1v6', 'tx0.1v2'):
        raise ValueError("Resolution must be one of 'gx1v6' or 'tx0.1v2'")

def pop_path(exp, res="gx1v6"):
    """
    Load native POP output for different experiments on Snellius
    (must be adapted for other personal runs).

    Input:
        exp: Experiment name
        res: Resolution ('gx1v6' or 'tx0.1v2')

    Returns:
        Path to the experiment parent folder
    """
    check_res(res)
    if exp == "default":
        if res == "gx1v6":
            return "/projects/0/prace_imau/prace_2013081679/pop/gx1v6/pop.B2000.gx1v6.qe_hosing.001"
        elif res == "tx0.1v2":
            return "/projects/0/prace_imau/prace_2013081679/pop/tx0.1v2/pop.B2000.tx0.1v2.qe_hosing.001"
    else:
        return f"/projects/0/prjs1105/oliver/pop/output/{res}/{exp}"

def load_pop(exp, res, year, month, dask=False):
    """
    Load monthly POP output

    Inputs:
        exp: Experiment name
        res: Resolution ('gx1v6' or 'tx0.1v2')
        year (int): Model year
        month (int): Model month (offset of file names will be corrected)
        dask (bool): Lazy loading using dask (recommended for high resolution) (default: False)
    
    Returns:
        xarray.Dataset
    """
    inpath = pop_path(exp, res)+"/tavg"
    file_ext = "t.x1_SAMOC_flux" if res=="gx1v6" else "t.t0.1_42l_nccs01"

    # File names are offset by one month
    if month == 12:
        file_path = f"{inpath}/{file_ext}.{(year+1):04d}01.nc"
    else:
        file_path = f"{inpath}/{file_ext}.{year:04d}{(month+1):02d}.nc"
    
    if dask:
        return xr.open_dataset(file_path, chunks={"k": 1})
    else:
        return xr.open_dataset(file_path)

def ym_string(year, month):
    return f"{year:04d}-{month:02d}"

def load_grid(res):
    """
    Load POP grid file
    """
    check_res(res)
    if res == "gx1v6":
        return xr.load_dataset("/home/omehling/models/pop/grid/grid_gx1v6.nc").rename({"nlat": "j", "nlon": "i"})
    else:
        return xr.load_dataset("/home/omehling/models/pop/grid/grid_tx0.1v2.nc").rename({"nlat": "j", "nlon": "i"})

def load_z(res):
    """
    Load depth

    Returns:
        z: Layer midpoints (in m)
        dz: Layer depths (in m)
    """
    check_res(res)
    if res == "gx1v6":
        depth_file 	= np.loadtxt("/home/omehling/models/pop/grid/in_depths.40.dat")
        depth		= depth_file[:, 1]
        layer		= depth_file[:, 0] / 100.0
    else:
        depth_file 	= np.loadtxt("/home/omehling/models/pop/grid/in_depths.42.dat")
        layer		= depth_file[:, 0] / 100.0
        depth = np.zeros(len(layer))
        for k in range(len(layer)):
            if k==0:
                depth[k] = depth[0]/2
            else:
                depth[k] = np.sum(depth[:k])+depth[k]/2

    z = xr.DataArray(depth, dims=["k"])
    dz = xr.DataArray(layer, dims=["k"])
    return z, dz

def mask_basin(grid, basin):
    if basin == "global":
        return grid["REGION_MASK"]>0
    
    basin_codes = {
        "atlantic": [6,8,11], # Atlantic + Labrador Sea + Hudson Bay
        "med": [7],
        "arctic": [9,10], # Nordic Seas + central Arctic Ocean
        "pacific": [2],
        "indian": [3,4], # Indian Ocean + Persian Gulf
        "southern_ocean": [1]
    }
    basin_codes["atlantic_arctic"] = basin_codes["atlantic"] + basin_codes["arctic"]
    basin_codes["indo_pacific"] = basin_codes["indian"] + basin_codes["pacific"]
    basins = list(basin_codes.keys())
    if basin not in basins:
        raise ValueError(f"Basin '{basin}' not supported, must be one of {basins}")
    
    mask = grid["REGION_MASK"]
    select = []
    for bnum in basin_codes[basin]:
        select.append((mask == bnum))
    return np.any(select)