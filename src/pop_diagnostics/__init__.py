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
        dV = (self.grid.TAREA/(100**2)*self.grid.dz).where(self.mask3D(mask))
        return - (S_anom*dV).sum()/S0
    
    def F_surf(self, data, mask):
        """
        Returns net surface freshwater flux in region mask, decomposed into components.
        """
        P = data.PRECIP.where(mask)
        E = data.EVAP.where(mask)
        R = data.RUNOFF.where(mask) # kg/m^2/s
        rest = data.S_WEAK_REST.where(mask)
        total = data.SFWF.where(mask) 
        melt = total - (P + E + R + rest)
        return [total, P, E, R, melt]
    
    def F_transport(self, data, mask=None, S0=35):
        if mask is None:
            mask = self.mask2D('Atlantic')
            
        S = data.SALT.where(self.mask3D(mask))*1000 # g/kg
        v = data.VVEL.where(self.mask3D(mask))/100 # m/s

        v_zonav = v.weighted(self.grid.DXU).mean(dim='i')
        v_secav = v_zonav.weighted(self.grid.dz).mean(dim='k')
        v_star = v - v_secav
        v_prime = v - v_zonav

        S_zonav = S.weighted(self.grid.DXT).mean(dim='i')
        S_secav = S_zonav.weighted(self.grid.dz).mean(dim='k')
        S_prime = S - S_zonav

        dx = self.grid.DXU/100
        dz = self.grid.dz

        # barotropic
        F_bt = - v_secav*(S_secav-S0)/S0*((dx*dz).where(self.mask3D(mask)).sum(dim=['i', 'k']))*1e-6 # Sv
        # overturning
        F_ov = - ((v_star*dx).sum(dim='i')*(S_zonav - S0)*dz).sum(dim='k')/S0*1e-6 # Sv
        # azimuthal
        F_az = - ((v_prime*S_prime*dx).sum(dim='i')*dz).sum(dim='k')/S0*1e-6 # Sv

        return F_bt, F_ov, F_az


     # Imported methods
     #from ._overturning import streamfunction_z, streamfunction_sigma