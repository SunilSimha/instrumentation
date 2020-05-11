from instrumentsim.ccd import noise, point_source as ps
import astropy.visualization as vis
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
import numpy as np

def circ_aper_photom(img:np.ndarray, x:float, y:float,
                     r:float, r_ann_in:float = 15,
                     r_ann_out:float = 20):
    """
    Measures the flux within a circular aperture
    and estimates the SNR using the
    local sky background flux estimated using
    an annulus. Assumes a sky dominated
    background level.
    Args
    ----
    img (ndarray): 2D image 
    x, y (float): location in pixels.
    r (float): radius of aperture in pixels
    r_ann_in, r_ann_out (float, optional): Sky annulus
        inner and outer radii in pixels.
    Returns
    -------
    phot_table (astropy Table): Summary table
        of aperture photometry results.
    """
    circ_aper = CircularAperture((x, y), r)
    sky_annulus = CircularAnnulus((x, y), r_ann_in, r_ann_out)
    apers = [circ_aper, sky_annulus]

    phot_table = aperture_photometry(img, apers)

    # Compute sky subtracted flux

    bkg_counts = phot_table['aperture_sum_1']/sky_annulus.area*circ_aper.area # This is the background flux estimated in the star aperture
    phot_table['object_flux'] = phot_table['aperture_sum_0']-bkg_counts

    phot_table['bkg_flux'] = bkg_counts

    # Compute SNR
    phot_table['SNR'] = phot_table['object_flux']/np.sqrt(phot_table['aperture_sum_0'])
    return phot_table

def mock_img(obj_counts:float, platescale:float,exptime:float = 600,
             imsize:tuple = (4096, 4112),
             x:float = None, y:float = None, fwhm:float = 1.4,
             bkg_flux:float = 150, dark:float = 3, rn:float = 3,
             sat_count:int = 200000,gaussianPSF=False):
    """
    Create a mock image of a point source
    on a CCD.
    Args
    ----
    obj_counts (float): Photon influx from
        object (photons/sec).
    platescale (float): arcsec/pixel at the
        image plane.
    exptime (float): exposuretime
    imsize (tuple, optional): size of the image
    x, y (float, optional): location of the point
        source. The object is placed at the
        image center by default.
    fwhm (float, optional): PSF FWHM in arcsec
    bkg_flux (float, optional): Sky background
        influx in photons/sec/arcsec^2.
    dark (float, optional): dark current in
        e-/pixel/hour.
    rn (float, optional): std.dev of read noise
        in e-
    sat_count (float, optional): Saturation
        level for each pixel. Counts beyond this
        are clipped.
    gaussianPSF (bool, optional): If true, the
        PSF is assumed to be gaussian.
    Returns
    -------
    img (ndarray): A mock CCD image.
    """

    # Prepare for image creation.
    if x is None:
        x = imsize[0]/2
    if y is None:
        y = imsize[1]/2

    # Convert FWHM to pixels
    fwhm_pix = fwhm/platescale

    # Compute object and sky levels.
    obj_lvl = obj_counts*exptime
    bkg_lvl = bkg_flux*platescale**2 # per pixel

    # Begin by creating object image
    if gaussianPSF:
        img = ps.add_gaussian_ps(x, y, fwhm_pix, obj_lvl, imsize)
    else:
        img = ps.add_ps(x, y, fwhm_pix, obj_lvl, imsize)

    # Add sky background
    bkg = noise.sky_bkg(bkg_lvl, imsize, exptime)

    # Add dark current
    bkg += noise.dark_img(dark, imsize, exptime)

    # Apply poisson noise
    img = noise.add_poisson_noise(img+bkg)

    # Clip saturated pixels
    img = noise.saturation_clip(img, sat_count)

    # Apply read noise
    img = noise.add_read_noise(img, rn)

    # Return
    return img, bkg

# Make a visulization wrapper
def draw_img(ax, fig, img, vmin=None, vmax = None,
             x = None, y = None, range = 50,
             cmap = 'hot', cblabel = None, cbticks = None,
             title = None, linstretch=False):
    """
    A wrapper for imshow
    """
    if linstretch:
        norm = None
    else:
        norm = vis.ImageNormalize(stretch=vis.LogStretch())
    im = ax.imshow(img, vmin = vmin, vmax = vmax, cmap = cmap, norm = norm)

    if x is None:
        x = img.shape[0]/2
    if y is None:
        y = img.shape[1]/2
    ax.set_xlim(x-range, x+range)
    ax.set_ylim(y-range, y+range)
    if cbticks is not None:
        cb = fig.colorbar(im, ax= ax, label = cblabel, ticks = cbticks, fraction = 0.045)
    else:
        cb = fig.colorbar(im, ax=ax, label = cblabel, fraction = 0.045)
    if title:
        ax.set_title(title)
    ax.grid(ls="--")
    return fig,ax