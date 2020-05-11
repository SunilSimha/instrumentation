# Simulate a point source image
# (before CCD effects are applied)

from photutils.aperture import CircularAperture

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
