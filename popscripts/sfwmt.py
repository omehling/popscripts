#!/usr/bin/env python
import xarray as xr
import pandas as pd
import numpy as np
import gsw_xarray as gsw

# Constants
rho0 = 1026 # sea water density [kg/m^3]
cp = 3985 # specific heat capacity [J/kg/K]

def swmt_inputs(pop_mon, p=0):
    sss = pop_mon["SALT"].isel(k=0).load()*1000
    sst = pop_mon["TEMP"].isel(k=0).load()
    sst_conservative = gsw.CT_from_pt(SA=sss, pt=sst)

    rho_surf, alpha_surf, beta_surf = gsw.rho_alpha_beta(SA=sss, CT=sst_conservative, p=p)
    f_heat = -alpha_surf/cp*pop_mon["SHF"].load()
    f_heat.name = "buoyancy_flux_due_to_shf"
    f_heat.attrs = {"long_name": "Surface buoyancy flux due to surface heat flux", "units": "kg/m^2/s"}
    f_fw = beta_surf*1000*(sss/1000/(1-sss/1000))*pop_mon["SFWF"].load()*-1
    f_fw.name = "buoyancy_flux_due_to_sfwf"
    f_fw.attrs = {"long_name": "Surface buoyancy flux due to surface freshwater flux", "units": "kg/m^2/s"}

    alpha_surf.name = "alpha"
    f_fw_prefac = beta_surf*1000*(sss/1000/(1-sss/1000))
    f_fw_prefac.name = "f_fw_prefac"

    f_surf = (f_heat+f_fw)
    f_surf.name = "buoyancy_flux"
    f_surf.attrs = {"long_name": "Surface buoyancy flux", "units": "kg/m^2/s"}

    if p==0 and "PD" in list(pop_mon.data_vars):
        pd_surf = pop_mon["PD"].isel(k=0).load()*1000
    else:
        pd_surf = rho_surf
    #pd_surf = pd_surf.where(pd_surf>=1000)
    pd_surf = pd_surf.where(pd_surf>pd_surf.min()+1)
    pd_surf.name = "density"
    pd_surf.attrs = {"long_name": "Surface density", "units": "kg/m^3"}

    return xr.merge([f_surf, f_heat, f_fw, pd_surf, alpha_surf, f_fw_prefac])

def swmt(buoyancy_flux, density, tarea, mask, sigmas):
    swmt_calc = []
    for i, sig in enumerate(sigmas):
        if i==0:
            sigm1 = sig-((sigmas[i+1]-sig)/2)
        if i>0:
            sigm1 = (sigmas[i-1]+sig)/2

        if i==len(sigmas)-1:
            sigp1 = sig+((sig-sigmas[i-1])/2)
        else:
            sigp1 = (sigmas[i+1]+sig)/2
        dsigma = sigp1-sigm1

        swmt_sig = (buoyancy_flux*tarea).where(
            (density>1000+sigm1) & (density<=1000+sigp1) & mask
        ).sum(["i","j"])/dsigma*1e-6 # in Sv
        swmt_calc.append(swmt_sig)

    swmt_xr = xr.concat(swmt_calc, dim="sigma")
    swmt_xr["sigma"] = sigmas
    swmt_xr.name = "sfwmt"

    return swmt_xr

def swmt_from_inputs(swmt_in, grid, mask, sigmas):
    return swmt(
        swmt_in["buoyancy_flux"],
        swmt_in["density"],
        grid["TAREA"]*1e-4,
        mask,
        sigmas
    )

def swmt_components_from_inputs(swmt_in, grid, mask, sigmas):
    pass

def volume_below_sigma(cell_volume, density, mask, sigmas):
    vol_calc = []
    for sig in sigmas:
        vol_sig = cell_volume.where(mask & (density>(sig+1000))).sum(["k","j","i"])
        vol_calc.append(vol_sig)

    volsig = xr.concat(vol_calc, dim="sigma")
    volsig["sigma"] = sigmas
    volsig.name = "volume_below_sigma"

    return volsig