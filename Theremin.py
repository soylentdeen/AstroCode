import scipy
import numpy
import SEDTools
import SpectralTools

def findSpectrumShift(x, flat, x_sm, y_sm):
    """
        This routine finds the wavelength and continuum shifts for a given
        wavelength window
    """
    window = scipy.where( (x > min(x_sm)) & (x < max(x_sm)) )[0]
    feature_x = x[window]
    model = scipy.interpolate.interpolate.interp1d(x_sm, y_sm, kind='linear',
            bounds_error = False)
    ycorr = scipy.correlate((1.0-flat[window]), (1.0-model(feature_x)), mode='full')
    xcorr = scipy.linspace(0, len(ycorr)-1, num=len(ycorr))

    fitfunc = lambda p,x: p[0]*scipy(-(x-p[1])**2.0/(2.0*p[2]**2.0)) + p[3]
    errfunc = lambda p,x,y: fitfunc(p,x) - y

    x_zoom = xcorr[len(ycorr)/2 - 3: len(ycorr)/2+5]
    y_zoom = ycorr[len(ycorr)/2 - 3: len(ycorr)/2+5]

    p_guess = [ycorr[len(ycorr)/2], len(ycorr)/2, 3.0, 0.0001]
    p1, success = scipy.optimize.leastsq(errfunc, p_guess, args = (x_zoom, y_zoom))

    fit = p1[0]*scipy.exp(-(x_zoom-p1[1])**2/(2.0*p1[2]**2)) + p1[3]

    xcorr = p1[0]
    nLags = xcorr-(len(window)-1.5)
    offset_computed = nLags*(feature_x[0]-feature_x[1])
    if abs(offset_computed) > 20:
        offset_computed = 0

    return offset_computed

def fitBestFitVeiling():
    """
        This routine finds the best fit veiling
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


