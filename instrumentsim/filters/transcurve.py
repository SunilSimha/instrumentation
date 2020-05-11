"""
A module to handle transmission curves.
All photometric relations are obtained from
https://arxiv.org/pdf/1407.6095.pdf
"""
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.constants import c, h
from astropy import units as u

from scipy.interpolate import interp1d
from scipy.integrate import simps
from pkg_resources import resource_filename

_vegatab = Table.read(resource_filename('instrumentsim','data/alpha_lyr_stis_010.fits'), hdu=1)

class Transmission_curve(object):
    """
    A master class for transmission
    curves.
    Args
    ----
    wave (Quantity array): a 1D wavelength array
        with astropy units.
    trans (array): a corresponding 1D array of
        the fraction of photons transmitted.
    """
    def __init__(self, wave, trans):
        assert len(wave)==len(trans), "Can't combine arrays of different lengths"
        self.wave = wave
        self.trans = trans
        # Interpolate the tranmission curve
        # in angstroms
        self.tcurve = interp1d(wave.to("AA").value, trans, bounds_error=False, fill_value=0, kind='cubic')
    
    def synthcounts(self, flam, lam):
        """
        Compute the average flux of photons
        in the band from a spectrum
        Args
        ----
        flam (Quantity array): Flux per unit
            wavelength interval
        lam (Quantity array): Wavelength array.
        Returns
        -------
        counts (Quantity): photon flux
            in counts/s/cm^2
        """
        assert len(flam)==len(lam), "Can't combine arrays of different lengths"

        # Work with arrays float instead of Quantities
        wave = lam.to('AA').value
        fflam = flam.to('erg*s**-1*cm**-2*AA**-1').value

        # Compute transmission
        t = self.tcurve(wave)
        # Integrate over wavelengths
        integ = simps(wave*fflam*t,wave)
        counts = integ/(h*c).to('AA*erg').value/(u.s*u.cm**2)
        return counts

    def _refinteg(self, wave, t, vega):
        """
        Helper function for computing
        a reference flux in magnitude
        calculations.
        """
        if vega:
            # Compute the vega flux in the band
            tt = self.tcurve(_vegatab['WAVELENGTH'])
            refinteg = simps(_vegatab['WAVELENGTH']*_vegatab['FLUX']*tt, _vegatab['WAVELENGTH'])
        else:
            # AB magnitude and match dimensions with integ
            refinteg = 3.631e-20*c.to('AA/s').value*simps(t/wave, wave)
        return refinteg

    def synthmag(self, flam, lam, vega=False):
        """
        Compute the magnitude from a spectrum
        Args
        ----
        flam (Quantity array): Flux per unit
            wavelength interval
        lam (Quantity array): Wavelength array.
        vega (bool, optional): If true, the
            magnitudes are computed in the Vega
            system.
        Returns
        -------
        mag (float): photometric magnitude
        """
        assert len(flam)==len(lam), "Can't combine arrays of different lengths"

        # Work with arrays float instead of Quantities
        wave = lam.to('AA').value
        fflam = flam.to('erg*s**-1*cm**-2*AA**-1').value

        # Compute transmission
        t = self.tcurve(wave)
        # Integrate over wavelengths
        integ = simps(wave*fflam*t,wave)

        refinteg = self._refinteg(wave, t, vega)
        # Return magnitude
        return -2.5*np.log10(integ/refinteg)
            
    def magtocounts(self, mag, vega=False):
        """
        Given a magnitude estimate
        in the band, compute the net
        photon count rate.
        Args
        ----
        mag (float): magnitude value
        vega (bool, optional): If true,
            the magnitude is assumed to
            be in the Vega system. AB is
            assumed otherwise.
        Returns
        -------
        counts (Quantity): photon arrival 
            rate per unit area (counts/s/cm^2).
        """
        refinteg = self._refinteg(self.wave.to('AA'), self.trans, vega)

        counts = 10**(-0.4*mag)*refinteg/(h*c).to('AA*erg').value/(u.s*u.cm**2)

        return counts

    def magtoflux(self, mag, vega=False):
        """
        Given a magnitude estimate
        in the band, compute the effective
        monochromatic flux density.
        Args
        ----
        mag (float): magnitude value
        vega (bool, optional): If true,
            the magnitude is assumed to
            be in the Vega system. AB is
            assumed otherwise.
        Returns
        -------
        flux (Quantity): Effective specific
            flux (erg/s/cm^2/AA).
        """
        refinteg = self._refinteg(self.wave, self.trans, vega)

        flux = 10**(-0.4*mag)*refinteg/simps(self.wave.to('AA')*self.trans, self.wave.to('AA'))*u.erg/u.s/u.cm**2/u.AA

        return flux