#!/usr/bin/env python
import xarray as xr
import pandas as pd
import numpy as np
import gsw_xarray as gsw
import joblib

# Constants
rho0 = 1026 # sea water density [kg/m^3]
cp = 3985 # specific heat capacity [J/kg/K]

def swmt_inputs(pop_mon):
    sss = pop_mon["SALT"].isel(k=0).load()*1000
    sst = pop_mon["TEMP"].isel(k=0).load()
    sst_conservative = gsw.CT_from_pt(SA=sss, pt=sst)

    rho_surf, alpha_surf, beta_surf = gsw.rho_alpha_beta(SA=sss, CT=sst_conservative, p=0)
    f_heat = alpha_surf/cp*pop_mon["SHF"].load()
    f_heat.name = "buoyancy_flux_due_to_shf"
    f_heat.attrs = {"long_name": "Surface buoyancy flux due to surface heat flux", "units": "kg/m^2/s"}
    f_fw = beta_surf*1000*(sss/1000/(1-sss/1000))*pop_mon["SFWF"].load()*-1
    f_fw.name = "buoyancy_flux_due_to_sfwf"
    f_fw.attrs = {"long_name": "Surface buoyancy flux due to surface freshwater flux", "units": "kg/m^2/s"}

    alpha_surf.name = "alpha"
    f_fw_prefac = beta_surf*1000*(sss/1000/(1-sss/1000))
    f_fw_prefac.name = "f_fw_prefac"

    #f_surf = (f_heat+f_fw)
    #f_surf.name = "buoyancy_flux"
    #f_surf.attrs = {"long_name": "Surface buoyancy flux", "units": "kg/m^2/s"}
    try:
        pd_surf = pop_mon["PD"].isel(k=0).load()*1000
    except KeyError:
        pd_surf = rho_surf
    pd_surf = pd_surf.where(pd_surf>=1000)
    pd_surf.name = "density"
    pd_surf.attrs = {"long_name": "Surface density", "units": "kg/m^3"}

    return xr.merge([f_heat, f_fw, pd_surf, alpha_surf, f_fw_prefac])

def swmt_from_inputs(swmt_in, mask, sigmas):
    pass    

def swmt_components_from_inputs(swmt_in, mask, sigmas):
    pass