# Simulate a point source image
# (before CCD effects are applied)

from photutils.aperture import CircularAperture
from astropy.modeling.models import Gaussian2D
import numpy as np

_fwhm2sig = 1/(2*np.sqrt(2*np.log(2)))

def add_ps(x:float,y:float,
           fwhm:float, counts:int,
           imsize:tuple):
    """
    Add a point source to an image.
    The PSF is assumed to be a flat
    circle.

    Args
    ----
    x, y (float): location in pixels.
    fwhm (float): FWHM of the PSF in pixels
    counts (float): total counts obtained
        in the span of an exposure.
    imsize (float tuple): Detector
        size in pixels
        (x_length, y_length).
    Returns
    -------
    ps_img (ndarray): image of point source
    """
    # initialise Aperture object
    ps = CircularAperture((x,y), fwhm/2)

    # Normalize by the area and scale by counts
    ps_img = ps.to_mask(method='exact').to_image(imsize)*counts/ps.area

    # return
    return ps_img

def add_gaussian_ps(x:float,y:float,
           fwhm:float, counts:int,
           imsize:tuple):
    """
    Add a point source to an image.
    The PSF is assumed to be a 2D gaussian
    with a diagonal covariance matrix
    and equal diagonal elements.

    Args
    ----
    x, y (float): location in pixels.
    fwhm (float): FWHM of the PSF in pixels
    counts (float): total counts obtained
        in the span of an exposure.
    imsize (float tuple): Detector
        size in pixels
        (x_length, y_length).
    Returns
    -------
    ps_img (ndarray): image of point source
    """
    sigma = fwhm*_fwhm2sig
    amplitude = counts/(2*np.pi*sigma**2)

    g = Gaussian2D(amplitude, x, y, sigma, sigma)

    xx, yy = np.mgrid[0:imsize[0], 0:imsize[1]]
    ps_img = g(yy,xx)
    return ps_img