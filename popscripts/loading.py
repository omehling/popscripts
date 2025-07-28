#!/usr/bin/env python
import xarray as xr
import numpy as np
import pandas as pd
from numba import njit

# GLOBAL CONFIG: data paths
grid_path = "/home/omehling/models/pop/grid"
exp_base_path = "/projects/0/prjs1105/oliver/pop/output"

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
        return f"{exp_base_path}/{res}/{exp}"

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

def add_time_dim(ds, year, month):
    time_ax = xr.date_range(ym_string(year, month), periods=1, freq="MS", calendar="365_day")
    return ds.expand_dims({"time": time_ax})

def load_grid(res):
    """
    Load POP grid file
    """
    check_res(res)
    if res == "gx1v6":
        return xr.load_dataset(grid_path+"/grid_gx1v6.nc").rename({"nlat": "j", "nlon": "i"})
    else:
        return xr.load_dataset(grid_path+"/grid_tx0.1v2.nc").rename({"nlat": "j", "nlon": "i"})

def load_z(res):
    """
    Load depth

    Returns:
        z: Layer midpoints (in m)
        dz: Layer depths (in m)
    """
    check_res(res)
    if res == "gx1v6":
        depth_file 	= np.loadtxt(grid_path+"/in_depths.40.dat")
        depth		= depth_file[:, 1]
        layer		= depth_file[:, 0] / 100.0
    else:
        depth_file 	= np.loadtxt(grid_path+"/in_depths.42.dat")
        layer		= depth_file[:, 0] / 100.0
        depth = np.zeros(len(layer))
        for k in range(len(layer)):
            if k==0:
                depth[k] = layer[0]/2
            else:
                depth[k] = np.sum(layer[:k])+layer[k]/2

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
    return xr.DataArray(np.any(select, axis=0), coords=mask.coords)

def load_straits(res, subset=None):
    """
    Import list of straits from `straits_def_<RES>.txt` in the directory `grid_path`.
    See example file in `data/` for the expected file format

    Args:
        res: Model resolution (gx1v6 or tx0.1v2)
        subset: subset of straits (default: None = include all straits)

    Returns:
        pandas.DataFrame
    """
    check_res(res)

    strait_list_import = pd.read_csv(grid_path+f"/straits_def_{res}.txt", header=None)
    strait_list = pd.DataFrame(index=range(len(strait_list_import)), columns=[
        "i_start", "i_stop", "j_start", "j_stop",
        "nr", "levels", "orientation", "strait"
    ])

    for i in range(len(strait_list_import)):
        list_cond = [item for item in strait_list_import[0][i].split(" ") if item != '']
        strait_list.iloc[i,:7] = list_cond[:7]
        strait_list.iloc[i,-1] = " ".join(list_cond[7:])

    strait_list.iloc[:,:6] = strait_list.iloc[:,:6].astype("int")

    if subset is None:
        return strait_list
    else:
        return strait_list[
            (strait_list.loc[:,"strait"].isin(subset)) |
            (strait_list.loc[:,"strait"].apply(lambda x: x[:-2]).isin(subset)) # with -X suffix
        ]

@njit
def dzu_partial_bottom(ktot, jtot, itot, kmu, depth_u, dz):
    """
    Cell depths (DZU) with partial bottom cells

    Args:
        ktot, jtot, itot: Size of dimensions (k,j,i)
        kmu (j,i): Input field KMU (index of deepest U-layer)
        depth_u (j,i): Input field HU (depth at U-points) in cm
        dz (k): Depth of layers in m
    
    Returns:
        numpy.array DZU (in m) with dimensions (k,j,i)
    """
    depth_u = depth_u/100
    zbot = np.cumsum(dz)
    
    cell_vol_np = np.zeros((ktot, jtot, itot))
    for j in range(jtot):
        for i in range(itot):
            kmax = int(kmu[j,i])-1 # 0-indexed
            if kmax == -1:
                continue
            for k in range(kmax):
                cell_vol_np[k,j,i] = dz[k]
            cell_vol_np[kmax,j,i] = dz[kmax]-(zbot[kmax]-depth_u[j,i])
    return cell_vol_np