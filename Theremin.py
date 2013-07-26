import scipy
import numpy
import SEDTools
import SpectralTools
import matplotlib.pyplot as pyplot

def findSpectrumShift(x, flat, x_sm, y_sm):
    """
    This routine finds the wavelength and continuum shifts for a given
    wavelength window
    """
    fig = pyplot.figure(0)
    ax=fig.add_axes([0.1, 0.1, 0.8, 0.8])
    window = scipy.where( (x > min(x_sm)) & (x < max(x_sm)) )[0]
    feature_x = x[window]
    fine_x = numpy.linspace(min(feature_x), max(feature_x), num=len(feature_x)*10.0)
    model = scipy.interpolate.interpolate.interp1d(x_sm, y_sm, kind='linear',bounds_error = False)
    observed = scipy.interpolate.interpolate.interp1d(feature_x, flat[window], kind='linear', bounds_error=False)
    ycorr = scipy.correlate((1.0-observed(fine_x)), (1.0-model(fine_x)), mode='full')
    xcorr = scipy.linspace(0, len(ycorr)-1, num=len(ycorr))

    #fitfunc = lambda p,x: p[0]*scipy.exp(-(x-p[1])**2.0/(2.0*p[2]**2.0)) + p[3]
    #errfunc = lambda p,x,y: fitfunc(p,x) - y

    x_zoom = xcorr[len(ycorr)/2 - 100: len(ycorr)/2+100]
    y_zoom = ycorr[len(ycorr)/2 - 100: len(ycorr)/2+100]
    max_index = numpy.argsort(y_zoom)[-1]
    offset_computed = (x_zoom[max_index] - len(xcorr)/2.0)/10.0*(feature_x[1]-feature_x[0])
    #ax.plot(x_zoom, y_zoom)
    #fig.show()
    #print wl_shift
    #raw_input()

    #p_guess = [ycorr[len(ycorr)/2], len(ycorr)/2, 3.0, 0.0001]
    #p1, success = scipy.optimize.leastsq(errfunc, p_guess, args = (x_zoom, y_zoom))

    #fit = p1[0]*scipy.exp(-(x_zoom-p1[1])**2/(2.0*p1[2]**2)) + p1[3]

    #xcorr = p1[0]
    #nLags = xcorr-(len(window)-1.5)
    #offset_computed = nLags*(feature_x[0]-feature_x[1])
    #if (abs(offset_computed) > 20):
    #    print 'Ha!', offset_computed
    #    offset_computed = 0
    #print asdf

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

def findContinuumPoints(x, y, **kwargs):
    """
       This routine uses normalized synthetic spectra to find regions of
       spectra close to the conintuum level.  The continuum points can then
       be used to compare observed spectra to the synthetic spectra
    """
    continuum_threshold = 0.01
    if kwargs.has_key("continuum_threshold"):
        continuum_threshold = kwargs["continuum_threshold"]

    continuum = scipy.where( abs(y-1.0) < continuum_threshold)[0]
    return continuum

def findBestFitVeiling():
    """
        This procedure should find the best fit veiling and continuum values
        for the fiducial value
    """
def interpolateModel(T, G, B):
    """
        This procedure should return an interpolated model
    """


