import scipy
import numpy
import SEDTools
import SpectralTools

def findSpectrumShift(x_window, flat, x_sm, y_sm):
    """
        This routine finds the wavelength and continuum shifts for a given
        wavelength window
    """

def binSyntheticSpectrum(spectrum, native_wl, new_wl):
    """
        This routine pixelates a synthetic spectrum, in effect simulating the 
        discrete nature of detector pixels.
    """
    retval = numpy.zeros(len(new_wl))
    for i in range(len(new_wl)-1):
        bm = scipy.where( (native_wl > new_wl[i]) & (
            native_wl <= new_wl[i+1]))[0]
        if (len(bm) > 0):
            num=scipy.integrate.simps(spectrum[bm], x=native_wl[bm])
            denom = max(native_wl[bm]) - min(native_wl[bm])
            retval[i] = num/denom
        else:
            retval[i] = retval[-1]

    bm = scipy.where(native_wl > new_wl[-1])[0]
    if len(bm) > 1:
        num = scipy.integrate.simps(spectrum[bm], x=native_wl[bm])
        denom = max(native_wl[bm]) - min(native_wl[bm])
        retval[-1] = num/denom
    else:
        retval[-1] = spectrum[bm]

    return retval

def findBestFitVeiling():
    """
        This procedure should find the best fit veiling and continuum values
        for the fiducial value
    """
def interpolateModel(T, G, B):
    """
        This procedure should return an interpolated model
    """


