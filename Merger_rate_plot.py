import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value
from astropy import constants as const
import scienceplots

boxsize = 75.0
omega_m = 0.2726
omega_l = 0.7274
h = 0.704
vol_comov_box = (boxsize/h)**3 #Mpc^3

def sample_fun(z):
    return z*2

def unit_comoving_vol(z):

    D_H = const.c.to('km/s').value/cosmo.H0.value
    return D_H * cosmo.comoving_transverse_distance(z).value**2 * cosmo.inv_efunc(z)

def total_merger_rate(z_bh,zmax = 4,zbinsize = 0.01,lgzbinsize=0.2,lgzmin=-3.0,lgzmax = 1.0):
    
    lgz_bh = np.log10(z_bh)
    dVcratio = np.array([unit_comoving_vol(z) 
                                  for z in z_bh ]) * 4*np.pi/vol_comov_box
    
    Nmrg_zhist,zbin_edges = np.histogram(z_bh,range=(0,zmax),bins=int(zmax/zbinsize))
    zbins = zbin_edges[:-1]+0.5*zbinsize
    dNmrgdz,tmp = np.histogram(z_bh,weights=dVcratio,bins=zbin_edges)
    
    dt_zbins = []

    for i in range(zbins.size):
        zatage = cosmo.age(zbins[i]-0.5*zbinsize)-cosmo.age(zbins[i]+0.5*zbinsize)
        dt_zbins.append(float(zatage/u.Gyr))

    dt_zbins = np.array(dt_zbins)
    
    dNmrg_dzdt = np.array([dNmrgdz[i]/dt_zbins[i]/10**9
                           for i in range(zbins.size)]) ## yr^-1 

    
    #merger_rate = np.sum(dNmrg_dzdt * zbinsize)

    merger_rate = np.trapz(dNmrg_dzdt, dx = zbinsize)
    
    return merger_rate


def merger_rate_log_plot(z_bh,zmax = 7,zbinsize = 0.2,lgzbinsize=0.2,lgzmin=-4.0,lgzmax = 1.0):
    
    lgz_bh = np.log10(z_bh)
    dVcratio = np.array([unit_comoving_vol(z) 
                                  for z in z_bh ]) * 4*np.pi/vol_comov_box
    
    Nmrg_zhist,zbin_edges = np.histogram(z_bh,range=(0,zmax),bins=int(zmax/zbinsize))
    Nmrg_zhist[Nmrg_zhist==0] = 1.0e-10
    zbins = zbin_edges[:-1]+0.5*zbinsize

    dNmrgdz,tmp = np.histogram(z_bh,weights=dVcratio,bins=zbin_edges)
    Nmrg = dNmrgdz * zbinsize
    dt_zbins = []

    for i in range(zbins.size):
        zatage = cosmo.age(zbins[i]-0.5*zbinsize)-cosmo.age(zbins[i]+0.5*zbinsize)
        dt_zbins.append(float(zatage/u.Gyr))

    dt_zbins = np.array(dt_zbins)

    dNmrg_dzdt = np.array([Nmrg[i]/dt_zbins[i]/10**9/vol_comov_box
                           for i in range(zbins.size)]) ## yr^-1 cMpc^-3
    
    #print("total merger rate (yr^-1 cMpc^-3): ",np.sum(dNmrg_dzdt * zbinsize))
    
    dNmrg_dzdt_allsky = np.array([dNmrgdz[i]/dt_zbins[i]/10**9
                           for i in range(zbins.size)]) ## yr^-1 

    merger_rate = np.sum(dNmrg_dzdt_allsky * zbinsize)
    #print("total all sky merger rate (yr^-1): ",merger_rate)
    
    # dNmrg_dzdtobs_allsky = np.array([dNmrgdz[i]/dt_zbins[i]/(1+zbins[i])/10**9
    #                        for i in range(zbins.size)]) ## yr^-1 
    # print("total 'observed' all-sky merger rate (yr^-1): ",np.sum(dNmrg_dzdtobs_allsky * zbinsize))
    

    #log d^2N/dlogz/dt plot
    
    Nmrg_lgzhist,lgzbin_edges = np.histogram(lgz_bh,range=(lgzmin,lgzmax),
                                             bins=int((lgzmax-lgzmin)/lgzbinsize))
    
    Nmrg_lgzhist[Nmrg_lgzhist==0] = 1.0e-10
    lgzbins = lgzbin_edges[:-1]+0.5*lgzbinsize
    Nmrg_lgzhist = Nmrg_lgzhist.astype('float64')
    

    dNmrgdlogz,tmp = np.histogram(lgz_bh,weights=dVcratio,bins=lgzbin_edges)
    
    dt_lgzbins = []
    for i in range(lgzbins.size):
        zatage = cosmo.age(10**(lgzbins[i]-0.5*lgzbinsize))-cosmo.age(10**(lgzbins[i]+0.5*lgzbinsize))
        dt_lgzbins.append(float(zatage/u.Gyr))
    dt_lgzbins = np.array(dt_lgzbins)

    #dNmrg_dlogzdt_allsky = np.array([dNmrgdlogz[i]*10**lgzbins[i]*np.log(10)/dt_lgzbins[i]/(1+10**lgzbins[i])/10**9 for i in range(lgzbins.size)])
    dNmrg_dlogzdt_allsky = np.array([dNmrgdlogz[i]*10**lgzbins[i]*np.log(10)/dt_lgzbins[i]/10**9 for i in range(lgzbins.size)])

    return merger_rate, lgzbins, dNmrg_dlogzdt_allsky
    
    

    