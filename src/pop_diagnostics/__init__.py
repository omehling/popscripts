"""
Python module: pop_diagnostics
"""

import xarray as xr
import gsw

class POP():
    def __init__(self, resolution='gx1v6', gridfile="../../data/pop_grid40.nc"):
        """
        Initialize the POP class.

        Keyword arguments
        -----------------
        - resolution: either 'gx1v6' (default, 1 degree) or 'tx0.1v2' (1/10 degree)
        - gridfile: filepath to the grid geometry file
        """

        grid = xr.load_dataset(gridfile).rename(
            {'nlon': 'i', 'nlat': 'j', 'z_t': 'k'}
        )
        self.grid = grid
        self.res = resolution

        if self.res not in ('gx1v6', 'tx0.1v2'):
            raise ValueError("Resolution must be one of 'gx1v6' or 'tx0.1v2'")

    def mask2D(self, label, latrange=(-90,90)):
        if label == 'Atlantic':
            _mask = self.grid.REGION_MASK.isin([6,8,9]).astype(int)
            return _mask.where(
                (self.grid.TLAT>=latrange[0]) & (self.grid.TLAT<=latrange[1])
                ).fillna(0)
     

    def mask3D(self, mask2D=None):
        land = self.grid.topo_mask3D.fillna(0)
        if mask2D is None:
            return land
        else:
            return land * mask2D

    def freshw_content(self, data, mask, S0=35):
        """
        Calculates the freshwater content in the region mask for reference salinity 35.
        Returns the freshwater content in units of m^3.
        """
        S_anom = data.SALT.where(self.mask3D(mask))*1000 - S0
        volumes = self.grid.TAREA/(100**2)*self.grid.dz
        return - (S_anom*volumes).sum()/S0
    
    def F_surf(self, data, mask):
        """
        Returns net surface freshwater flux in region mask, decomposed into components.
        """
        P = data.PRECIP
        E = data.EVAP
        R = data.RUNOFF # kg/m^2/s
        rest = data.S_WEAK_REST
        melt = data.SFWF - (P + E + R + rest)
        return [data.SFWF, P, E, R, melt]
    
    def F_transport(self, data, mask, latidx=85, S0=35):
        S = data.SALT.where(self.mask3D(mask)).isel(j=latidx)*1000 # g/kg
        v = data.VVEL.where(self.mask3D(mask)).isel(j=latidx)/100 # m/s
        
        v_hat = v.weighted(self.grid.DXU.isel(j=latidx))


     # Imported methods
     #from ._overturning import streamfunction_z, streamfunction_sigma