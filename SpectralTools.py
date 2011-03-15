import scipy.signal
import scipy.interpolate
import scipy.optimize
import numpy
import pyfits

def resample(x, y, R):
    ''' This routine convolves a given spectrum to a resolution R'''

    subsample = 16.0

    xstart = x[0]
    xstop = x[-1]

    newx = [xstart]
    while newx[-1] < xstop:
        stepsize = newx[-1]/(R*subsample)
        newx.append(newx[-1]+stepsize)

    newx = numpy.array(newx)

    f = scipy.interpolate.interpolate.interp1d(x, y, bounds_error=False)
    newy = f(newx)
    const = numpy.ones(len(newx))

    xk = numpy.array(range(4.0*subsample))
    yk = numpy.exp(-(xk-(2.0*subsample))**2.0/(subsample**2.0/(4.0*numpy.log(2.0))))
    
    result = scipy.signal.convolve(newy, yk, mode ='valid')
    normal = scipy.signal.convolve(const, yk, mode = 'valid')

    bm = numpy.isfinite(result)
    return newx[len(xk)/2.0:-len(xk)/2.0], result[bm]/normal[bm]


def read_2col_spectrum(filename):
    '''
    Reads in a spectrum from a text file in a 2 column format.

    column 1: wavelength
    column 2: flux
    '''
    data = open(filename).read().split('\n')
    
    x = []
    y = []
    
    for line in data:
        l = line.split()
        if len(l) == 2:
            x.append(float(l[0]))
            y.append(float(l[1]))
            
    x = numpy.array(x)
    y = numpy.array(y)
    
    return x, y

def read_fits_spectrum(filename):
    hdulist = pyfits.open(filename, ignore_missing_end=True)
    hdr = hdulist[0].header
    dat = hdulist[0].data

    wl = dat[0]
    fl = dat[1]
    dfl = dat[2]

    return (wl, fl, dfl)

def fit_gaussians(x, y, linecenters, R):

    params = []
    strength = -0.05
    for line in linecenters:
        fwhm = line/R
        params.append(strength)      #Strength
        params.append(line)          #Line Center
        params.append(fwhm)          #FWHM

    def fitfunc(pars, xpts):
        retval = numpy.ones(len(xpts))
        for i in range(len(linecenters)):
            k = i*3
            for j in range(len(xpts)):
                retval[j] += pars[k]*numpy.exp(-(xpts[j]-pars[k+1])**2.0/(pars[k+2]))
        return retval

    def errfunc(pars, xpts, ypts):
        return numpy.abs(fitfunc(pars, xpts) - ypts)
        
    pfit = scipy.optimize.leastsq(errfunc, params, args=(x, y))

    coeffs = pfit[0]

    fit = numpy.ones(len(x))
    for i in range(len(linecenters)):
        k = i*3
        for j in range(len(x)):
            fit[j] += coeffs[k]*numpy.exp(-(x[j]-coeffs[k+1])**2.0/(coeffs[k+2]))

    return coeffs, fit
