# Functions for simulating noise and CCD effects
import numpy as np

def sky_bkg(flux:float, imsize:tuple,
            exptime:float, randomize:bool=False):
    """
    Simulate sky background
    Args
    ----
    flux (float): sky flux in 
        photons/s/pixel
    imsize (float tuple): Detector
        size in pixels
        (x_length, y_length).
    exptime (float): exposure time
        in seconds
    randomize (bool): If true, adds
        poisson noise.
    Returns
    -------
    sky_img (float array): Image
        of sky background with
        poisson noise.
    """
    # initialize
    sky_img = flux*exptime + np.zeros(imsize)
    
    # Add poisson noise
    if randomize:
        sky_img = np.random.poisson(sky_img)

    # return
    return sky_img

def dark_img(count_rate:float,
            imsize:tuple, exptime:float,
            randomize:bool = False):
    """
    Simulate dark current
    Args
    ----
    count_rate (float): dark charge accumulation
        in e-/hour
    imsize (float tuple): Detector
        size in pixels
        (x_length, y_length).
    exptime (float): exposure time
        in seconds
    Returns
    -------
    sky_img (float array): Image
        of sky background with
        poisson noise.
    """
    # initialize
    dark_img = count_rate*exptime/3600 + np.zeros(imsize)
    
    # Add poisson noise
    if randomize:
        dark_img = np.random.poisson(dark_img)

    # return
    return dark_img

def add_read_noise(img:np.ndarray, rn:float):
    """
    Add gaussian read noise
    to each pixel.
    Args
    ----
    img (float 2D array): image without
        read noise.
    rn (float): read noise standard deviation
    Returns
    -------
    noisy_img (float 2D array): image with read
        noise.
    """

    # Using the floor function because reading
    # adds or removes integer number of electrons.
    return np.floor(np.random.normal(img, rn))

def add_poisson_noise(img:np.ndarray):
    """
    Add poisson fluctuations
    to the image.
    Args
    ----
    img (float ndarray): 2D image without
        poisson noise
    Returns
    ------
    noisy_img (float ndarray): image
        with simulated poisson noise.
    """
    return np.random.poisson(img)

def saturation_clip(img:np.ndarray, sat_lvl:float):
    """
    Clip image at the specified saturation
    level.
    Args
    ----
    img (ndarray): 2D image
    sat_lvl (float): Saturation level beyond which
        counts are clipped.
    Returns
    -------
    sat_img (ndarray): Clipped 2D image
    """

    sat_img = np.where(img >= sat_lvl, sat_lvl, img)
    return sat_img
